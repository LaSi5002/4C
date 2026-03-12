// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors_service.hpp"

#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_mat_inelastic_defgrad_factors.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <boost/graph/visitors.hpp>

#include <algorithm>
#include <array>
#include <iomanip>
#include <string>
#include <utility>


FOUR_C_NAMESPACE_OPEN

using namespace Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

namespace
{
  // input for verifying optimal interpolation factors
  struct InputVerifyOptimalInterpolationFactors
  {
    const unsigned int gp_;
    const double reference_val_;
    const double elast_pred_val_;
    const double plast_pred_val_;
  };


  // determine and verify optimal interpolation factors
  double determine_optimal_interpolation_factors(
      InputVerifyOptimalInterpolationFactors input_verify_optimal_interpolation_factors,
      std::string id_for_val)
  {
    // set numerical tolerance for equality of numbers
    const double numerical_tol{1.0e-8};

    double optimal_xi_for_val = 0.0;

    const double delta_val_reference_elast =
        input_verify_optimal_interpolation_factors.reference_val_ -
        input_verify_optimal_interpolation_factors.elast_pred_val_;
    const double delta_val_plast_elast =
        input_verify_optimal_interpolation_factors.plast_pred_val_ -
        input_verify_optimal_interpolation_factors.elast_pred_val_;
    if (std::abs(delta_val_plast_elast) > numerical_tol)
    {
      optimal_xi_for_val = delta_val_reference_elast / delta_val_plast_elast;

      // project back to bounding box [0, 1]
      if (optimal_xi_for_val > 1.0)
      {
// warning
#ifdef DISPLAY_WARNINGS
        std::cout << "Optimal interpolation factor for " << id_for_val << ", gp "
                  << input_verify_optimal_interpolation_factors.gp_ << ": "
                  << input_verify_optimal_interpolation_factors.reference_val_
                  << " leads to optimal interpolation factor " << optimal_xi_for_val
                  << " for the bounds "
                  << input_verify_optimal_interpolation_factors.elast_pred_val_
                  << "(elastic predictor, 0) and "
                  << input_verify_optimal_interpolation_factors.plast_pred_val_
                  << "(plastic predictor, 1)" << std::endl;
#endif


        // set back to the higher bound
        optimal_xi_for_val = 1.0;

// log management strategy
#ifdef DISPLAY_WARNINGS
        std::cout << "Setting value back to higher bound (plastic predictor, 1)" << std::endl;
#endif
      }
      else if (optimal_xi_for_val < 0.0)
      {
// warning
#ifdef DISPLAY_WARNINGS
        std::cout << "Optimal interpolation factor for " << id_for_val << ", gp "
                  << input_verify_optimal_interpolation_factors.gp_ << ": "
                  << input_verify_optimal_interpolation_factors.reference_val_
                  << " leads to optimal interpolation factor " << optimal_xi_for_val
                  << " for the bounds "
                  << input_verify_optimal_interpolation_factors.elast_pred_val_
                  << "(elastic predictor, 0) and "
                  << input_verify_optimal_interpolation_factors.plast_pred_val_
                  << "(plastic predictor, 1)" << std::endl;
#endif


        // set back to the lower bound
        optimal_xi_for_val = 0.0;

// log management strategy
#ifdef DISPLAY_WARNINGS
        std::cout << "Setting value back to lower bound (elastic predictor, 0)" << std::endl;
#endif
      }
    }
    else
    {
      // we leave the value at the lower bound (even if the reference
      // value does not equal the common value at the bounds)

      if (std::abs(delta_val_reference_elast) > numerical_tol)
      {
// warning
#ifdef DISPLAY_WARNINGS
        std::cout << " Value for " << id_for_val << ", gp "
                  << input_verify_optimal_interpolation_factors.gp_ << ": "
                  << input_verify_optimal_interpolation_factors.reference_val_
                  << " does not equal the value of the bounds "
                  << input_verify_optimal_interpolation_factors.elast_pred_val_
                  << "(elastic predictor, 0) and "
                  << input_verify_optimal_interpolation_factors.plast_pred_val_
                  << "(plastic predictor, 1)" << std::endl;


        // log management strategy
        std::cout
            << "Setting optimal interpolation factor as the lower bound (elastic predictor, 0)"
            << std::endl;
#endif
      }
    }

    return optimal_xi_for_val;
  }

  // Local Newton Guess Interpolation: determine whether elastic stretch eigenvector rotation
  // shall be interpolated
  bool lngi_interpolate_elastic_stretch_eigenvect_rot(
      const LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvectRotType
          eigenvect_rot_type,
      const LocalNewtonGuessInterpolation::DefgradType defgrad_type)
  {
    switch (eigenvect_rot_type)
    {
      case LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvectRotType::
          elastic_predictor:
      {
        return (defgrad_type != LocalNewtonGuessInterpolation::DefgradType::elastic_defgrad);
      }
      default:
      {
        FOUR_C_THROW(
            "For plastic predictor eigenvector rotation type {}, we don't yet know whether the "
            "eigenvector rotation should be interpolated in the LNGI",
            eigenvect_rot_type);
      }
    }
  }

  // Local Newton Guess Interpolation: determine whether rotation
  // shall be interpolated
  bool lngi_interpolate_rot(
      const LocalNewtonGuessInterpolation::PlasticPredictorRotationType rot_type,
      const LocalNewtonGuessInterpolation::DefgradType defgrad_type)
  {
    switch (rot_type)
    {
      case LocalNewtonGuessInterpolation::PlasticPredictorRotationType::elastic_predictor:
      {
        return (defgrad_type != LocalNewtonGuessInterpolation::DefgradType::elastic_defgrad);
      }
      default:
      {
        FOUR_C_THROW(
            "For plastic predictor rotation type {}, we don't yet know whether the "
            "rotation should be interpolated in the LNGI",
            rot_type);
      }
    }
  }



  // Local Newton Guess Interpolation: determine which deformation gradient should be interpolated
  // (elastic deformation gradient or inverse plastic deformation gradient), depending on the
  // constructed plastic predictor
  LocalNewtonGuessInterpolation::DefgradType get_defgrad_type_to_be_interpolated(
      const LocalNewtonGuessInterpolation::PlasticPredictorRotationType rot_type)
  {
    switch (rot_type)
    {
      case LocalNewtonGuessInterpolation::PlasticPredictorRotationType::elastic_predictor:
      {
        return LocalNewtonGuessInterpolation::DefgradType::elastic_defgrad;
      }
      default:
      {
        FOUR_C_THROW("Defgrad type cannot be determined yet for plastic predictor rotation type {}",
            rot_type);
      }
    }
  }


}  // namespace



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::string
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::get_detailed_error_message_for_error_type(
    ErrorType err_type)
{
  switch (err_type)
  {
    case ErrorType::negative_plastic_strain:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: negative plastic strain!";
    case ErrorType::overflow_error:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: overflow error related to "
             "the evaluation of the plastic strain increment!";
    case ErrorType::no_plastic_incompressibility:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: plastic incompressibility "
             "not satisfied!";
    case ErrorType::failed_solution_linear_system_lnl:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the Local Newton Loop failed!";
    case ErrorType::failed_determ_line_search_step:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: could not determine a "
             "suitable line search parameter!";
    case ErrorType::no_convergence_local_newton:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Local Newton Loop did not "
             "converge for the given loop settings!";
    case ErrorType::singular_jacobian:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: singular Jacobian after "
             "converged Local Newton Loop, which does not allow for the analytical evaluation "
             "of the linearization!";
    case ErrorType::failed_solution_analytic_linearization:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the analytical linearization failed";
    case ErrorType::failed_computation_flow_resistance:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the flow resistance of the viscoplasticity law";
    case ErrorType::failed_computation_flow_resistance_derivs:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the derivatives of the flow resistance of the viscoplasticity law";
    case ErrorType::failed_matrix_log_evaluation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
             "matrix logarithm or its derivative with respect to the argument";
    case ErrorType::failed_matrix_exp_evaluation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
             "matrix exponential or its derivative with respect to the argument";
    case ErrorType::failed_right_cg_interpolation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in interpolating "
             "the right Cauchy-Green deformation tensor";
    case ErrorType::under_yield_surface:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: we are 'under' the yield "
             "surface, sigma < sigma_yield!";
    default:
      FOUR_C_THROW("to_string(ErrorType): You should not be here!");
  }
}

Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ConstNonMatTensors::ConstNonMatTensors()
{  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;
  Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);

  // set constant non-material tensors

  // 3x3 identity
  id3x3_.update(1.0, id3x3, 0.0);

  // Voigt stress form of 3x3 identity
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      id3x3_, id6x1_);

  // symmetric identity four tensor
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      id4_6x6_, 1.0, id3x3, id3x3, 0.0);

  // deviatoric operator
  Core::LinAlg::FourTensor<3> dev_op_four_tensor =
      Core::LinAlg::setup_deviatoric_projection_tensor<3>();
  Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, dev_op_four_tensor);
  dev_op_ = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // identity four tensor
  id4_9x9_.clear();
  Core::LinAlg::FourTensorOperations::add_non_symmetric_product(1.0, id3x3_, id3x3_, id4_9x9_);

  // 10x10 identity
  id10x10_.clear();
  for (int i = 0; i < 10; ++i) id10x10_(i, i) = 1.0;
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ConstMatTensors::
    set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m)
{
  // get instance of constant non-material tensors
  const auto& const_non_mat_tensors = ConstNonMatTensors::instance();

  // set material-dependent tensors (fiber orientation)

  // structural tensor
  mm_.multiply_nt(1.0, m, m, 0.0);

  // deviatoric part of the structural tensor
  double tr_mm_ = mm_(0, 0) + mm_(1, 1) + mm_(2, 2);
  mm_dev_.update(1.0, mm_, -1.0 / 3.0 * tr_mm_, const_non_mat_tensors.id3x3_);

  // dyadic product of structural tensors
  Core::LinAlg::Matrix<6, 1> mm_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm_, mm_V);
  mm_dyad_mm_.multiply_nt(1.0, mm_V, mm_V, 0.0);

  // dyadic product of deviatoric structural tensor with the structural tensor
  Core::LinAlg::Matrix<6, 1> mm_dev_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm_dev_, mm_dev_V);
  mm_dev_dyad_mm_.multiply_nt(1.0, mm_dev_V, mm_V, 0.0);

  // dyadic product of identity with the structural tensor
  id_dyad_mm_.multiply_nt(1.0, const_non_mat_tensors.id6x1_, mm_V, 0.0);

  // sum of identity with the structural tensor
  id_plus_mm_.update(1.0, const_non_mat_tensors.id3x3_, 1.0, mm_, 0.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
    PredictorDefgradDecomposition::PredictorDefgradDecomposition(
        const Core::LinAlg::Matrix<3, 3>& defgrad_elast_pred,
        const Core::LinAlg::Matrix<3, 3>& defgrad_plast_pred,
        std::optional<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>>
            spectral_pairs_ref)
    : specific_defgrad_elast_pred_(defgrad_elast_pred),
      specific_defgrad_plast_pred_(defgrad_plast_pred)
{
  //  perform polar decomposition of defgrad within elastic predictor
  Core::LinAlg::Matrix<3, 3> material_stretch_matrix_elast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> rotation_matrix_elast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> eigenval_matrix_elast_pred{Core::LinAlg::Initialization::zero};
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_elast_pred;
  Core::LinAlg::matrix_3x3_polar_decomposition(defgrad_elast_pred, rotation_matrix_elast_pred,
      material_stretch_matrix_elast_pred, eigenval_matrix_elast_pred, spectral_pairs_elast_pred);

  //  perform polar decomposition of defgrad within plastic predictor
  Core::LinAlg::Matrix<3, 3> material_stretch_matrix_plast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> rotation_matrix_plast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> eigenval_matrix_plast_pred{Core::LinAlg::Initialization::zero};
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_plast_pred;
  Core::LinAlg::matrix_3x3_polar_decomposition(defgrad_plast_pred, rotation_matrix_plast_pred,
      material_stretch_matrix_plast_pred, eigenval_matrix_plast_pred, spectral_pairs_plast_pred);

  // collect all spectral pairs (elastic and plastic predictors) and use
  // this for ordering and alignment: we will rewrite them back to their
  // original form afterwards
  std::vector<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>> all_spectral_pairs{
      spectral_pairs_elast_pred, spectral_pairs_plast_pred};

  // set reference locations for interpolation
  Core::LinAlg::Matrix<1, 1> ref_loc_elast_pred;
  ref_loc_elast_pred(0, 0) = 0.0;
  Core::LinAlg::Matrix<1, 1> ref_loc_plast_pred;
  ref_loc_plast_pred(0, 0) = 1.0;
  std::vector<Core::LinAlg::Matrix<1, 1>> all_ref_locs{ref_loc_elast_pred, ref_loc_plast_pred};

  // align spectral pairs of reference (elastic predictor) suitably
  Core::LinAlg::align_eigenpairs_of_base_matrix(all_spectral_pairs, all_ref_locs, 0);

  // order eigenpairs (plastic predictor) with respect to the reference (elastic predictor)
  if (spectral_pairs_ref.has_value())
  {
    Core::LinAlg::order_eigenpairs_wrt_reference(spectral_pairs_ref.value(), all_spectral_pairs[1]);
    // save spectral pairs within designated objects
    spectral_pairs_elast_pred_ = spectral_pairs_ref.value();
  }
  else
  {
    Core::LinAlg::order_eigenpairs_wrt_reference(all_spectral_pairs[0], all_spectral_pairs[1]);
    // save spectral pairs within designated objects
    spectral_pairs_elast_pred_ = all_spectral_pairs[0];
  }
  spectral_pairs_plast_pred_ = all_spectral_pairs[1];

  // save eigenvalues
  lambda_elast_pred_ = {spectral_pairs_elast_pred_[0].first, spectral_pairs_elast_pred_[1].first,
      spectral_pairs_elast_pred_[2].first};
  lambda_plast_pred_ = {spectral_pairs_plast_pred_[0].first, spectral_pairs_plast_pred_[1].first,
      spectral_pairs_plast_pred_[2].first};
  log_lambda_elast_pred_ = {std::log(spectral_pairs_elast_pred_[0].first),
      std::log(spectral_pairs_elast_pred_[1].first), std::log(spectral_pairs_elast_pred_[2].first)};
  log_lambda_plast_pred_ = {std::log(spectral_pairs_plast_pred_[0].first),
      std::log(spectral_pairs_plast_pred_[1].first), std::log(spectral_pairs_plast_pred_[2].first)};

  // save eigenvector rotation matrices and vectors
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      Qmat_elast_pred_(i, j) = spectral_pairs_elast_pred_[i].second(j);
      Qmat_plast_pred_(i, j) = spectral_pairs_plast_pred_[i].second(j);
    }
  }
  Qmat_plast_pred_rel_.multiply_tn(1.0, Qmat_elast_pred_, Qmat_plast_pred_, 0.0);
  Qvec_plast_pred_rel_ = Core::LinAlg::calc_rot_vect_from_rot_matrix(Qmat_plast_pred_rel_);

  // save rotation matrices and vectors
  Rmat_elast_pred_ = rotation_matrix_elast_pred;
  Rmat_plast_pred_ = rotation_matrix_plast_pred;
  Rmat_plast_pred_rel_.multiply_tn(1.0, Rmat_elast_pred_, Rmat_plast_pred_, 0.0);
  Rvec_plast_pred_rel_ = Core::LinAlg::calc_rot_vect_from_rot_matrix(Rmat_plast_pred_rel_);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
    LocalNewtonGuessInterpolation(const double k_scan, const unsigned int max_num_reinterp,
        const PlasticPredictorElasticStretchEigenvalType elastic_stretch_eigenval_type,
        const PlasticPredictorElasticStretchEigenvectRotType elastic_stretch_eigenvect_rot_type,
        const PlasticPredictorRotationType rot_type, const double min_interp_interval)
    : k_scan_(k_scan),
      num_of_lngi_(0),
      max_num_lngi_(max_num_reinterp),
      min_interp_interval_(min_interp_interval),
      guess_inv_plast_defgrad_{Core::LinAlg::Matrix<10, 1>{Core::LinAlg::Initialization::zero}},
      plast_pred_elast_stretch_eigenval_type_(elastic_stretch_eigenval_type),
      plast_pred_elast_stretch_eigenvect_rot_type_(elastic_stretch_eigenvect_rot_type),
      plast_pred_rot_type_(rot_type),
      defgrad_type_(get_defgrad_type_to_be_interpolated(rot_type)),
      interpolate_elastic_stretch_eigenvect_rot_(lngi_interpolate_elastic_stretch_eigenvect_rot(
          elastic_stretch_eigenvect_rot_type, defgrad_type_)),
      interpolate_rot_(lngi_interpolate_rot(rot_type, defgrad_type_))

{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> id3x3{Core::LinAlg::Initialization::zero};
  for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;

  // initialize predictor decompositions for first GP (using the identity matrices as
  // dummy matrices for
  // now)
  curr_pred_decomp_specific_defgrad_.resize(1, {PredictorDefgradDecomposition(id3x3, id3x3)});
  last_pred_decomp_specific_defgrad_.resize(1, {PredictorDefgradDecomposition(id3x3, id3x3)});

  // initialize component interpolators for first GP (using zero-values)
  all_component_interp_lambda_1_.resize(1, {ComponentInterpolator<1>{.current_xi_ = {0.0},
                                               .last_xi_ = {0.0},
                                               .optimal_xi_ = {0.0},
                                               .xi_l_ = {0.0},
                                               .xi_u_ = {0.0}}});
  all_component_interp_lambda_2_.resize(1, {ComponentInterpolator<1>{.current_xi_ = {0.0},
                                               .last_xi_ = {0.0},
                                               .optimal_xi_ = {0.0},
                                               .xi_l_ = {0.0},
                                               .xi_u_ = {0.0}}});
  all_component_interp_rel_eigenvect_rot_.resize(
      1, {ComponentInterpolator<3>{.current_xi_ = {0.0, 0.0, 0.0},
             .last_xi_ = {0.0, 0.0, 0.0},
             .optimal_xi_ = {0.0, 0.0, 0.0},
             .xi_l_ = {0.0, 0.0, 0.0},
             .xi_u_ = {0.0, 0.0, 0.0}}});
};

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::setup(
    const unsigned int num_gp)
{
  // setup predictor decompositions with the right number of Gauss points
  curr_pred_decomp_specific_defgrad_.resize(num_gp, curr_pred_decomp_specific_defgrad_[0]);
  last_pred_decomp_specific_defgrad_.resize(num_gp, last_pred_decomp_specific_defgrad_[0]);

  // setup component interpolators with the right number of Gauss points
  all_component_interp_lambda_1_.resize(num_gp, all_component_interp_lambda_1_[0]);
  all_component_interp_lambda_2_.resize(num_gp, all_component_interp_lambda_2_[0]);
  all_component_interp_rel_eigenvect_rot_.resize(
      num_gp, all_component_interp_rel_eigenvect_rot_[0]);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
    pre_evaluate(const unsigned int gp,
        const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_elast_pred,
        const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_plast_pred,
        const Core::LinAlg::Matrix<3, 3>& defgrad)
{
  // reset bounds for to 0.0, 1.0 for all component interpolators
  all_component_interp_lambda_1_[gp].xi_l_ = {0.0};
  all_component_interp_lambda_1_[gp].xi_u_ = {1.0};

  all_component_interp_lambda_2_[gp].xi_l_ = {0.0};
  all_component_interp_lambda_2_[gp].xi_u_ = {1.0};

  all_component_interp_rel_eigenvect_rot_[gp].xi_l_ = {0.0, 0.0, 0.0};
  all_component_interp_rel_eigenvect_rot_[gp].xi_u_ = {1.0, 1.0, 1.0};

  // clear initial guess for inverse plastic defgrad, and the number of
  // performed interpolations
  guess_inv_plast_defgrad_.clear();
  num_of_lngi_ = 0;

  // decompose the deformation gradients involved in the elastic and plastic predictors
  perform_predictor_decomposition(
      gp, inv_plastic_defgrad_elast_pred, inv_plastic_defgrad_plast_pred, defgrad);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::update()
{
  // set last <- current for the component interpolators
  for (size_t gp = 0; gp < all_component_interp_lambda_1_.size(); ++gp)
  {
    all_component_interp_lambda_1_[gp].last_xi_ = all_component_interp_lambda_1_[gp].current_xi_;
    all_component_interp_lambda_2_[gp].last_xi_ = all_component_interp_lambda_2_[gp].current_xi_;
    if (interpolate_elastic_stretch_eigenvect_rot_)
    {
      all_component_interp_rel_eigenvect_rot_[gp].last_xi_ =
          all_component_interp_rel_eigenvect_rot_[gp].current_xi_;
    }
  }

  last_pred_decomp_specific_defgrad_ = curr_pred_decomp_specific_defgrad_;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::pack(
    Core::Communication::PackBuffer& data) const
{
  Core::Communication::add_to_pack(data, all_component_interp_lambda_1_);
  Core::Communication::add_to_pack(data, all_component_interp_lambda_2_);
  Core::Communication::add_to_pack(data, all_component_interp_rel_eigenvect_rot_);
  for (auto p : last_pred_decomp_specific_defgrad_)
  {
    p.pack(data);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id3x3{Core::LinAlg::Initialization::zero};
  for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;



  Core::Communication::extract_from_pack(buffer, all_component_interp_lambda_1_);
  Core::Communication::extract_from_pack(buffer, all_component_interp_lambda_2_);
  Core::Communication::extract_from_pack(buffer, all_component_interp_rel_eigenvect_rot_);
  last_pred_decomp_specific_defgrad_.resize(
      all_component_interp_lambda_1_.size(), {PredictorDefgradDecomposition(id3x3, id3x3)});
  for (auto p : last_pred_decomp_specific_defgrad_)
  {
    p.unpack(buffer);
  }

  // set current <- last for component interpolators
  for (size_t gp = 0; gp < all_component_interp_lambda_1_.size(); ++gp)
  {
    all_component_interp_lambda_1_[gp].current_xi_ = all_component_interp_lambda_1_[gp].last_xi_;
    all_component_interp_lambda_2_[gp].current_xi_ = all_component_interp_lambda_2_[gp].last_xi_;
    if (interpolate_elastic_stretch_eigenvect_rot_)
    {
      all_component_interp_rel_eigenvect_rot_[gp].current_xi_ =
          all_component_interp_rel_eigenvect_rot_[gp].last_xi_;
    }
  }

  // initialize predictor decompositions for first GP (using the identity matrices as
  // dummy matrices for
  // now)
  curr_pred_decomp_specific_defgrad_.resize(
      all_component_interp_lambda_1_.size(), {PredictorDefgradDecomposition(id3x3, id3x3)});
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
    perform_predictor_decomposition(const unsigned int gp,
        const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_elast_pred,
        const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_plast_pred,
        const Core::LinAlg::Matrix<3, 3>& defgrad)
{
  // perform required decomposition depending on defgrad type
  if (defgrad_type_ == DefgradType::elastic_defgrad)
  {
    // compute elastic deformation gradients for both predictors
    Core::LinAlg::Matrix<3, 3> elastic_defgrad_elast_pred{Core::LinAlg::Initialization::zero};
    elastic_defgrad_elast_pred.multiply_nn(1.0, defgrad, inv_plastic_defgrad_elast_pred, 0.0);
    Core::LinAlg::Matrix<3, 3> elastic_defgrad_plast_pred{Core::LinAlg::Initialization::zero};
    elastic_defgrad_plast_pred.multiply_nn(1.0, defgrad, inv_plastic_defgrad_plast_pred, 0.0);

    // perform decompositions of the elastic deformation gradients
    curr_pred_decomp_specific_defgrad_[gp] =
        PredictorDefgradDecomposition(elastic_defgrad_elast_pred, elastic_defgrad_plast_pred);
  }
  else if (defgrad_type_ == DefgradType::inv_plastic_defgrad)
  {
    // perform decompositions of the inverse plastic deformation gradients
    curr_pred_decomp_specific_defgrad_[gp] = PredictorDefgradDecomposition(
        inv_plastic_defgrad_elast_pred, inv_plastic_defgrad_plast_pred);
  }
  else
  {
    FOUR_C_THROW("Unsupported deformation gradient type {}", defgrad_type_);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
    LocalNewtonGuessInterpolation::interpolate_inv_plastic_defgrad(const unsigned int gp,
        const Core::LinAlg::Matrix<3, 3>& defgrad, const InterpolationPoint& interp_point,
        const Core::LinAlg::Matrix<3, 3>& inv_defgrad)
{
  // declare interpolated inverse plastic deformation gradient
  Core::LinAlg::Matrix<3, 3> inv_plastic_defgrad{Core::LinAlg::Initialization::zero};

  // interpolate rotation vector through linear interpolation (eigenvector rotation) -> use
  // elastic predictor (= 0 relative rotation) if this should not be interpolated
  Core::LinAlg::Matrix<3, 1> rel_eigenvect_rot_vect{Core::LinAlg::Initialization::zero};
  if (interpolate_elastic_stretch_eigenvect_rot_)
  {
    for (unsigned int i = 0; i < 3; ++i)
    {
      rel_eigenvect_rot_vect(i) = interp_point.xi_rel_eigenvect_rot_[i] *
                                  curr_pred_decomp_specific_defgrad_[gp].Qvec_plast_pred_rel_(i);
    }
  }

  // get relative eigenvector (rotation) matrix associated with the rotation
  // vector (relative with respect to the eigenvector matrix within the
  // elastic predictor)
  Core::LinAlg::Matrix<3, 3> rel_eigenvect_rot_matrix =
      Core::LinAlg::calc_rot_matrix_from_rot_vect(rel_eigenvect_rot_vect);
  // compute eigenvector (rotation) matrix from the relative rotation
  // vector and the reference, i.e., the eigenvector (rotation) matrix
  // within the elastic predictor
  Core::LinAlg::Matrix<3, 3> eigenvect_rot_matrix{Core::LinAlg::Initialization::zero};
  eigenvect_rot_matrix.multiply_nn(
      1.0, curr_pred_decomp_specific_defgrad_[gp].Qmat_elast_pred_, rel_eigenvect_rot_matrix, 0.0);

  // calculate eigenvalue matrix assuming plastic incompressibility (det
  // = 1) --> set only first two eigenvalues, the third one is set based on the
  // considered deformation gradient type
  Core::LinAlg::Matrix<3, 3> eigenvalue_matrix{Core::LinAlg::Initialization::zero};

  // --> eigenvalue interpolation

  // compute weights
  const double w_lambda_1_elast_pred = 1.0 - interp_point.xi_lambda_1_;
  const double w_lambda_1_plast_pred = interp_point.xi_lambda_1_;

  eigenvalue_matrix(0, 0) = std::exp(
      w_lambda_1_elast_pred * curr_pred_decomp_specific_defgrad_[gp].log_lambda_elast_pred_[0] +
      w_lambda_1_plast_pred * curr_pred_decomp_specific_defgrad_[gp].log_lambda_plast_pred_[0]);
  eigenvalue_matrix(1, 1) = std::exp(
      w_lambda_1_elast_pred * curr_pred_decomp_specific_defgrad_[gp].log_lambda_elast_pred_[1] +
      w_lambda_1_plast_pred * curr_pred_decomp_specific_defgrad_[gp].log_lambda_plast_pred_[1]);

  /*
  #ifdef DEBUG_PRED_ADAPT
    std::cout << "Interpolating inverse plastic defgrad: " << std::endl;
    std::cout << "--> defgrad_type: " << EnumTools::enum_name(defgrad_type_) << std::endl;
    std::cout << "--> lambda_1: elast_pred: "
              << all_pred_decomp_specific_defgrad_[gp].lambda_elast_pred_[0]
              << "; plast_pred: " << all_pred_decomp_specific_defgrad_[gp].lambda_plast_pred_[0]
              << "; current_xi: " << all_component_interp_lambda_1_[gp].current_xi_[0] <<
  std::endl; std::cout << "--> lambda_2: elast_pred: "
              << all_pred_decomp_specific_defgrad_[gp].lambda_elast_pred_[1]
              << "; plast_pred: " << all_pred_decomp_specific_defgrad_[gp].lambda_plast_pred_[1]
              << "; current_xi: " << all_component_interp_lambda_2_[gp].current_xi_[0] <<
  std::endl; std::cout << "--> lambda_3: elast_pred: "
              << all_pred_decomp_specific_defgrad_[gp].lambda_elast_pred_[2]
              << "; plast_pred: " << all_pred_decomp_specific_defgrad_[gp].lambda_plast_pred_[2]
              << std::endl;
    std::cout << "--> Qmat_elast_pred: " << std::endl;
    all_pred_decomp_specific_defgrad_[gp].Qmat_elast_pred_.print(std::cout);
    std::cout << "--> Qmat_plast_pred: " << std::endl;
    all_pred_decomp_specific_defgrad_[gp].Qmat_plast_pred_.print(std::cout);
    std::cout << "--> Qvec_plast_pred_rel_: " << std::endl;
    all_pred_decomp_specific_defgrad_[gp].Qvec_plast_pred_rel_.print(std::cout);
    std::cout << "--> eigenvect_rot_matrix: " << std::endl;
    eigenvect_rot_matrix.print(std::cout);
    std::cout << "inv_defgrad (relevant for defgrad_type: elastic_defgrad): " << std::endl;
    inv_defgrad.print(std::cout);
  #endif
  */


  // different treatment based on the utilized deformation gradient type for the
  // interpolation
  if (defgrad_type_ == DefgradType::elastic_defgrad)
  {
    // the elastic deformation gradient has the determinant of the full
    // deformation gradient
    eigenvalue_matrix(2, 2) =
        defgrad.determinant() / (eigenvalue_matrix(0, 0) * eigenvalue_matrix(1, 1));

    // build elastic deformation gradient
    Core::LinAlg::Matrix<3, 3> elastic_defgrad{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> lambda_Q{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> QT_lambda_Q{Core::LinAlg::Initialization::zero};
    lambda_Q.multiply_nn(1.0, eigenvalue_matrix, eigenvect_rot_matrix, 0.0);
    QT_lambda_Q.multiply_tn(1.0, eigenvect_rot_matrix, lambda_Q, 0.0);
    elastic_defgrad.multiply_nn(1.0, curr_pred_decomp_specific_defgrad_[gp].Rmat_elast_pred_,
        QT_lambda_Q, 0.0);  // both R rotation matrices can be taken, since they should be the
                            // same! This is already checked previously!

    // compute inverse inelastic deformation gradient
    inv_plastic_defgrad.multiply_nn(1.0, inv_defgrad, elastic_defgrad, 0.0);
  }
  else if (defgrad_type_ == DefgradType::inv_plastic_defgrad)
  {
    // the plastic deformation gradient has the determinant 1 -> plastic
    // incompressibility assumed!
    eigenvalue_matrix(2, 2) = 1.0 / (eigenvalue_matrix(0, 0) * eigenvalue_matrix(1, 1));

    // build inverse plastic deformation gradient
    Core::LinAlg::Matrix<3, 3> lambda_Q{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> QT_lambda_Q{Core::LinAlg::Initialization::zero};
    lambda_Q.multiply_nn(1.0, eigenvalue_matrix, eigenvect_rot_matrix, 0.0);
    QT_lambda_Q.multiply_tn(1.0, eigenvect_rot_matrix, lambda_Q, 0.0);
    inv_plastic_defgrad.multiply_nn(
        1.0, curr_pred_decomp_specific_defgrad_[gp].Rmat_plast_pred_, QT_lambda_Q, 0.0);
  }
  else
  {
    FOUR_C_THROW("Unsupported deformation gradient type {}", defgrad_type_);
  }


  return inv_plastic_defgrad;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
    adapt_interpolation_intervals(const unsigned int gp, const ErrorType eval_err_type)
{
  // collect all current interpolation factors
  std::vector<double> all_current_xi;

  if (interpolate_elastic_stretch_eigenvect_rot_)
  {
    all_current_xi = {all_component_interp_lambda_1_[gp].current_xi_[0],
        all_component_interp_lambda_2_[gp].current_xi_[0],
        all_component_interp_rel_eigenvect_rot_[gp].current_xi_[0],
        all_component_interp_rel_eigenvect_rot_[gp].current_xi_[1],
        all_component_interp_rel_eigenvect_rot_[gp].current_xi_[2]};
  }
  else
  {
    all_current_xi = {all_component_interp_lambda_1_[gp].current_xi_[0],
        all_component_interp_lambda_2_[gp].current_xi_[0]};
  }


  if (eval_err_type == ErrorType::no_errors)
  {
    return;
  }
  else if (eval_err_type ==
           ErrorType::under_yield_surface)  // check whether we are "under" the yield surface:
  // if the plastic strain rate is 0.0 (the adapted predictor maps the stress
  // "under" the yield surface): set \f$ \xi_{\text{curr}} \leftarrow
  // \xi_{\text{curr}} + \xi_{\text{user}} ( 1.0 - \xi_{\text{curr}})
  // \f$
  {
    // get maximum out of the interpolation factors
    const double max_current_xi = *std::max_element(all_current_xi.begin(), all_current_xi.end());

    // adapt the upper bounds of the interpolation factors
    set_upper_bound_interp_point(
        gp, InterpolationPoint{.xi_lambda_1_ = max_current_xi,
                .xi_lambda_2_ = max_current_xi,
                .xi_rel_eigenvect_rot_ = {max_current_xi, max_current_xi, max_current_xi}});
  }
  else  // there is "too much" plastic strain rate -> leads to overflow error
  {
    // get minimum out of the interpolation factors
    const double min_current_xi = *std::min_element(all_current_xi.begin(), all_current_xi.end());

    // adapt the lower bounds of the interpolation factors
    set_lower_bound_interp_point(
        gp, InterpolationPoint{.xi_lambda_1_ = min_current_xi,
                .xi_lambda_2_ = min_current_xi,
                .xi_rel_eigenvect_rot_ = {min_current_xi, min_current_xi, min_current_xi}});
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
    adapt_interpolation_parameters(const unsigned int gp)
{
  // adapt interpolation parameters based on the current interval
  set_curr_interp_point(gp, add_interpolation_points(1 - k_scan_, get_lower_bound_interp_point(gp),
                                k_scan_, get_upper_bound_interp_point(gp)));
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
    InterpolationPoint
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
        compute_optimal_interp_factors(
            const PredictorDefgradDecomposition& predictor_defgrad_decomposition, unsigned int gp,
            const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_solution,
            const Core::LinAlg::Matrix<3, 3>& defgrad)
{
  // considered solution matrix for optimal interpolation factors: different treatment based on
  // the considered deformation gradient type
  Core::LinAlg::Matrix<3, 3> solution_defgrad{Core::LinAlg::Initialization::zero};
  if (defgrad_type_ == DefgradType::elastic_defgrad)
  {
    solution_defgrad.multiply_nn(1.0, defgrad, inv_plastic_defgrad_solution, 0.0);
  }
  else if (defgrad_type_ == DefgradType::inv_plastic_defgrad)
  {
    solution_defgrad = inv_plastic_defgrad_solution;
  }
  else
  {
    FOUR_C_THROW(
        "Unsupported deformation gradient type {} for initial guess interpolation", defgrad_type_);
  }

  // decompose solution, using the elastic predictor as reference as in the
  // interpolation routine --> we do this without rerotating the elastic predictor since this was
  // our base during interpolation (i.e., the eigenvectors have the same ordering as in our
  // current predictor decomposition)
  PredictorDefgradDecomposition solution_defgrad_decomposition{
      predictor_defgrad_decomposition.specific_defgrad_elast_pred_, solution_defgrad,
      predictor_defgrad_decomposition.spectral_pairs_elast_pred_};

  // consistency check: if the elastic defgrad is interpolated, the R-rotation
  // of the solution must be the same as in the elastic predictor (and the
  // plastic predictor)
  if (defgrad_type_ == DefgradType::elastic_defgrad)
  {
    const double rel_rot_norm = solution_defgrad_decomposition.Rvec_plast_pred_rel_.norm2();
    FOUR_C_ASSERT_ALWAYS(rel_rot_norm < 1.0e-8,
        "GP {}: Inconsistency when determining the optimal interpolation factors. The relative "
        "R-rotation "
        "is not 0, but [{}, {}, {}]",
        gp, solution_defgrad_decomposition.Rvec_plast_pred_rel_(0),
        solution_defgrad_decomposition.Rvec_plast_pred_rel_(1),
        solution_defgrad_decomposition.Rvec_plast_pred_rel_(2));
  }


  // determine the optimal interpolation factors
  const double optimal_xi_lambda_1 = determine_optimal_interpolation_factors(
      InputVerifyOptimalInterpolationFactors{.gp_ = gp,
          .reference_val_ = solution_defgrad_decomposition.lambda_plast_pred_[0],
          .elast_pred_val_ = curr_pred_decomp_specific_defgrad_[gp].lambda_elast_pred_[0],
          .plast_pred_val_ = curr_pred_decomp_specific_defgrad_[gp].lambda_plast_pred_[0]},
      "lambda 1");
  const double optimal_xi_lambda_2 = determine_optimal_interpolation_factors(
      InputVerifyOptimalInterpolationFactors{.gp_ = gp,
          .reference_val_ = solution_defgrad_decomposition.lambda_plast_pred_[1],
          .elast_pred_val_ = curr_pred_decomp_specific_defgrad_[gp].lambda_elast_pred_[1],
          .plast_pred_val_ = curr_pred_decomp_specific_defgrad_[gp].lambda_plast_pred_[1]},
      "lambda 2");

  std::array<double, 3> optimal_xi_rel_eigenvect_rot{0.0, 0.0, 0.0};
  if (interpolate_elastic_stretch_eigenvect_rot_)
  {
    for (unsigned int i = 0; i < 3; ++i)
    {
      optimal_xi_rel_eigenvect_rot[i] = determine_optimal_interpolation_factors(
          InputVerifyOptimalInterpolationFactors{.gp_ = gp,
              .reference_val_ = solution_defgrad_decomposition.Qvec_plast_pred_rel_(i),
              .elast_pred_val_ = 0.0,
              .plast_pred_val_ = curr_pred_decomp_specific_defgrad_[gp].Qvec_plast_pred_rel_(i)},
          "eigenvector rotation vector, component " + std::to_string(i));
    }
  }


  return InterpolationPoint{.xi_lambda_1_ = optimal_xi_lambda_1,
      .xi_lambda_2_ = optimal_xi_lambda_2,
      .xi_rel_eigenvect_rot_ = optimal_xi_rel_eigenvect_rot};
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::reset()
{
  t_ = 0.0;
  substep_counter_ = 0;
  curr_dt_ = 0.0;
  time_step_halving_counter_ = 0;
  total_num_of_substeps_ = 0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    init_csv_writer()
{
  // get structure discretization
  std::shared_ptr<Core::FE::Discretization> structure_dis =
      Global::Problem::instance()->get_dis("structure");

  // check whether we are using a single processor! (no implementation for multiple processors
  // yet, and also not really required)
  int my_rank = Core::Communication::my_mpi_rank(structure_dis->get_comm());
  FOUR_C_ASSERT_ALWAYS(my_rank == 0,
      "InelasticDefgradTransvIsotropElastViscoplast: No implementation of time integration "
      "output "
      "for multiple processors");

  // create csv_writer and register its columns
  csv_writer_.emplace(
      my_rank, *Global::Problem::instance()->output_control_file(), "timint_output");
  csv_writer_->register_data_vector("Eval. steps (LNL)", 1, 16);
  csv_writer_->register_data_vector("Eval. iterations (LNL)", 1, 16);
  csv_writer_->register_data_vector("Eval. reinterpolations (LNGI)", 1, 16);
  csv_writer_->register_data_vector("Eval. iterations (LNGI)", 1, 16);
  csv_writer_->register_data_vector("Eval. iterations (LNGI reinterpolation)", 1, 16);
  csv_writer_->register_data_vector("Eval. line searches (LNL)", 1, 16);
  csv_writer_->register_data_vector("Eval. iterations (line search)", 1, 16);
  csv_writer_->register_data_vector("Eval. time (RMA)", 1, 16);
  csv_writer_->register_data_vector("Eval. time (LNGI preparation)", 1, 16);
  csv_writer_->register_data_vector("Total steps (LNL)", 1, 16);
  csv_writer_->register_data_vector("Total iterations (LNL)", 1, 16);
  csv_writer_->register_data_vector("Total reinterpolations (LNGI)", 1, 16);
  csv_writer_->register_data_vector("Total iterations (LNGI)", 1, 16);
  csv_writer_->register_data_vector("Total iterations (LNGI reinterpolation)", 1, 16);
  csv_writer_->register_data_vector("Total line searches (LNL)", 1, 16);
  csv_writer_->register_data_vector("Total iterations (line search)", 1, 16);
  csv_writer_->register_data_vector("Total time (RMA)", 1, 16);
  csv_writer_->register_data_vector("Total time (LNGI preparation)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (lambda 1) of GP 0 of Ele 0 (last global iteration)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (lambda 2) of GP 0 of Ele 0 (last global iteration)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (q component 1) of GP 0 of Ele 0 (last global iteration)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (q component 2) of GP 0 of Ele 0 (last global iteration)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (q component 3) of GP 0 of Ele 0 (last global iteration)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (lambda 1) of GP 0 of Ele 0 (maximum over all global "
      "iterations)",
      1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (lambda 2) of GP 0 of Ele 0 (maximum over all global "
      "iterations)",
      1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (q component 1) of GP 0 of Ele 0 (maximum over all global "
      "iterations)",
      1, 16);

  csv_writer_->register_data_vector(
      "Interpolation factor (q component 2) of GP 0 of Ele 0 (maximum over all global "
      "iterations)",
      1, 16);

  csv_writer_->register_data_vector(
      "Interpolation factor (q component 3) of GP 0 of Ele 0 (maximum over all global "
      "iterations)",
      1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (lambda 1) of GP 0 of Ele 0 (optimal)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (lambda 2) of GP 0 of Ele 0 (optimal)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (q component 1) of GP 0 of Ele 0 (optimal)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (q component 2) of GP 0 of Ele 0 (optimal)", 1, 16);
  csv_writer_->register_data_vector(
      "Interpolation factor (q component 3) of GP 0 of Ele 0 (optimal)", 1, 16);
  csv_writer_->register_data_vector(
      "LNL Residual: Interpolation factor of GP 0 of Ele 0 (optimal)", 1, 16);
  /*
  for (ErrorType err_type : magic_enum::enum_values<ErrorType>())
  {
    csv_writer_->register_data_vector(
        "Eval. Error " + std::string(magic_enum::enum_name(err_type)), 1, 16);
    csv_writer_->register_data_vector(
        "Total Error " + std::string(magic_enum::enum_name(err_type)), 1, 16);
  }*/
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    reset()
{
  num_iters_and_steps_.eval_num_of_LNL_steps_ = 0;
  num_iters_and_steps_.eval_num_of_iters_ = 0;
  num_iters_and_steps_.eval_num_of_lngi_reinterp_ = 0;
  num_iters_and_steps_.eval_num_of_lngi_iters_ = 0;
  num_iters_and_steps_.eval_num_of_reinterp_iters_ = 0;
  num_iters_and_steps_.eval_num_of_line_search_ = 0;
  num_iters_and_steps_.eval_num_of_line_search_iters_ = 0;
  timers_.eval_teuchos_timer_rma_.reset();
  timers_.eval_teuchos_timer_lngi_preparation_.reset();
  time_measurements_.eval_time_rma_ = 0;
  time_measurements_.eval_time_lngi_preparation_ = 0;
  eval_error_map_ = {
      {ErrorType::negative_plastic_strain, 0},
      {ErrorType::overflow_error, 0},
      {ErrorType::no_plastic_incompressibility, 0},
      {ErrorType::failed_solution_linear_system_lnl, 0},
      {ErrorType::failed_determ_line_search_step, 0},
      {ErrorType::no_convergence_local_newton, 0},
      {ErrorType::singular_jacobian, 0},
      {ErrorType::failed_solution_analytic_linearization, 0},
      {ErrorType::failed_matrix_log_evaluation, 0},
      {ErrorType::failed_matrix_exp_evaluation, 0},
      {ErrorType::under_yield_surface, 0},
  };
  num_iters_and_steps_.eval_num_of_alpha_neq_1_ = 0;
  num_iters_and_steps_.eval_num_of_alpha_neq_1_last_iter_ = 0;
  lngi_factors_.curr_lngi_factor_lambda_1_ = -1.0;
  lngi_factors_.curr_lngi_factor_lambda_2_ = -1.0;
  lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_0_ = -1.0;
  lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_1_ = -1.0;
  lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_2_ = -1.0;
  lngi_factors_.curr_max_lngi_factor_lambda_1_ = -1.0;
  lngi_factors_.curr_max_lngi_factor_lambda_2_ = -1.0;
  lngi_factors_.curr_max_lngi_factor_eigenvect_rot_comp_0_ = -1.0;
  lngi_factors_.curr_max_lngi_factor_eigenvect_rot_comp_1_ = -1.0;
  lngi_factors_.curr_max_lngi_factor_eigenvect_rot_comp_2_ = -1.0;
  lngi_factors_.optimal_lngi_factor_lambda_1_ = -1.0;
  lngi_factors_.optimal_lngi_factor_lambda_2_ = -1.0;
  lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_0_ = -1.0;
  lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_1_ = -1.0;
  lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_2_ = -1.0;
  lngi_factors_.lnl_res_optimal_lngi_factor_ = -1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    update_total()
{
  num_iters_and_steps_.total_num_of_LNL_steps_ += num_iters_and_steps_.eval_num_of_LNL_steps_;
  num_iters_and_steps_.total_num_of_iters_ += num_iters_and_steps_.eval_num_of_iters_;
  num_iters_and_steps_.total_num_of_lngi_reinterp_ +=
      num_iters_and_steps_.eval_num_of_lngi_reinterp_;
  num_iters_and_steps_.total_num_of_lngi_iters_ += num_iters_and_steps_.eval_num_of_lngi_iters_;
  num_iters_and_steps_.total_num_of_reinterp_iters_ +=
      num_iters_and_steps_.eval_num_of_reinterp_iters_;
  num_iters_and_steps_.total_num_of_line_search_ += num_iters_and_steps_.eval_num_of_line_search_;
  num_iters_and_steps_.total_num_of_line_search_iters_ +=
      num_iters_and_steps_.eval_num_of_line_search_iters_;
  num_iters_and_steps_.total_num_of_alpha_neq_1_ += num_iters_and_steps_.eval_num_of_alpha_neq_1_;
  num_iters_and_steps_.total_num_of_alpha_neq_1_last_iter_ +=
      num_iters_and_steps_.eval_num_of_alpha_neq_1_last_iter_;
  time_measurements_.total_time_rma_ += time_measurements_.eval_time_rma_;
  time_measurements_.total_time_lngi_preparation_ += time_measurements_.eval_time_lngi_preparation_;
  for (const auto& [error_type, error_count] : eval_error_map_)
  {
    total_error_map_[error_type] += error_count;
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    write_to_csv()
{
  // output data
  std::map<std::string, std::vector<double>> output_data;
  output_data["Eval. steps (LNL)"] = {
      static_cast<double>(num_iters_and_steps_.eval_num_of_LNL_steps_)};
  output_data["Total steps (LNL)"] = {
      static_cast<double>(num_iters_and_steps_.total_num_of_LNL_steps_)};
  output_data["Eval. iterations (LNL)"] = {
      static_cast<double>(num_iters_and_steps_.eval_num_of_iters_)};
  output_data["Total iterations (LNL)"] = {
      static_cast<double>(num_iters_and_steps_.total_num_of_iters_)};
  output_data["Eval. reinterpolations (LNGI)"] = {
      static_cast<double>(num_iters_and_steps_.eval_num_of_lngi_reinterp_)};
  output_data["Total reinterpolations (LNGI)"] = {
      static_cast<double>(num_iters_and_steps_.total_num_of_lngi_reinterp_)};
  output_data["Eval. iterations (LNGI)"] = {
      static_cast<double>(num_iters_and_steps_.eval_num_of_lngi_iters_)};
  output_data["Total iterations (LNGI)"] = {
      static_cast<double>(num_iters_and_steps_.total_num_of_lngi_iters_)};
  output_data["Eval. iterations (LNGI reinterpolation)"] = {
      static_cast<double>(num_iters_and_steps_.eval_num_of_reinterp_iters_)};
  output_data["Total iterations (LNGI reinterpolation)"] = {
      static_cast<double>(num_iters_and_steps_.total_num_of_reinterp_iters_)};
  output_data["Eval. iterations (line search)"] = {
      static_cast<double>(num_iters_and_steps_.eval_num_of_line_search_iters_)};
  output_data["Total iterations (line search)"] = {
      static_cast<double>(num_iters_and_steps_.total_num_of_line_search_iters_)};
  output_data["Eval. line searches (LNL)"] = {
      static_cast<double>(num_iters_and_steps_.eval_num_of_line_search_)};
  output_data["Total line searches (LNL)"] = {
      static_cast<double>(num_iters_and_steps_.total_num_of_line_search_)};
  output_data["Eval. time (RMA)"] = {static_cast<double>(time_measurements_.eval_time_rma_)};
  output_data["Total time (RMA)"] = {static_cast<double>(time_measurements_.total_time_rma_)};
  output_data["Eval. time (LNGI preparation)"] = {
      static_cast<double>(time_measurements_.eval_time_lngi_preparation_)};
  output_data["Total time (LNGI preparation)"] = {
      static_cast<double>(time_measurements_.total_time_lngi_preparation_)};
  /*
  output_data["Eval. # of times: alpha neq 1 (all LNL iters)"] = {
      static_cast<double>(eval_num_of_alpha_neq_1)};
  output_data["Eval. # of times: alpha neq 1 (last LNL iter)"] = {
      static_cast<double>(eval_num_of_alpha_neq_1_last_iter)};
  output_data["Eval. # of first LNL iter. convergences"] = {
      static_cast<double>(eval_num_of_first_iter_convergences)};
  output_data["Total # of times: alpha neq 1 (all LNL iters)"] = {
      static_cast<double>(total_num_of_alpha_neq_1)};
  output_data["Total # of times: alpha neq 1 (last LNL iter)"] = {
      static_cast<double>(total_num_of_alpha_neq_1_last_iter)};
  output_data["Total # of first LNL iter. convergences"] = {
      static_cast<double>(total_num_of_first_iter_convergences)};

  for (ErrorType err_type : magic_enum::enum_values<ErrorType>())
  {
    output_data["Eval. Error " + std::string(magic_enum::enum_name(err_type))] = {
        static_cast<double>(eval_error_map_[err_type])};
    output_data["Total Error " + std::string(magic_enum::enum_name(err_type))] = {
        static_cast<double>(total_error_map_[err_type])};
  }
*/


  // initial guess interpolation factors
  output_data
      ["Interpolation factor (lambda 1) of GP 0 of Ele 0 (last global "
       "iteration)"] = {static_cast<double>(lngi_factors_.curr_lngi_factor_lambda_1_)};
  output_data
      ["Interpolation factor (lambda 2) of GP 0 of Ele 0 (last global "
       "iteration)"] = {static_cast<double>(lngi_factors_.curr_lngi_factor_lambda_2_)};
  output_data
      ["Interpolation factor (q component 1) of GP 0 of Ele 0 (last global "
       "iteration)"] = {static_cast<double>(lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_0_)};
  output_data
      ["Interpolation factor (q component 2) of GP 0 of Ele 0 (last global "
       "iteration)"] = {static_cast<double>(lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_1_)};
  output_data
      ["Interpolation factor (q component 3) of GP 0 of Ele 0 (last global "
       "iteration)"] = {static_cast<double>(lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_2_)};
  output_data
      ["Interpolation factor (lambda 1) of GP 0 of Ele 0 (maximum over all global "
       "iterations)"] = {static_cast<double>(lngi_factors_.curr_max_lngi_factor_lambda_1_)};
  output_data
      ["Interpolation factor (lambda 2) of GP 0 of Ele 0 (maximum over all global "
       "iterations)"] = {static_cast<double>(lngi_factors_.curr_max_lngi_factor_lambda_2_)};
  output_data
      ["Interpolation factor (q component 1) of GP 0 of Ele 0 (maximum over all global "
       "iterations)"] = {
          static_cast<double>(lngi_factors_.curr_max_lngi_factor_eigenvect_rot_comp_0_)};
  output_data
      ["Interpolation factor (q component 2) of GP 0 of Ele 0 (maximum over all global "
       "iterations)"] = {
          static_cast<double>(lngi_factors_.curr_max_lngi_factor_eigenvect_rot_comp_1_)};
  output_data
      ["Interpolation factor (q component 3) of GP 0 of Ele 0 (maximum over all global "
       "iterations)"] = {
          static_cast<double>(lngi_factors_.curr_max_lngi_factor_eigenvect_rot_comp_2_)};
  output_data["Interpolation factor (lambda 1) of GP 0 of Ele 0 (optimal)"] = {
      static_cast<double>(lngi_factors_.optimal_lngi_factor_lambda_1_)};
  output_data["Interpolation factor (lambda 2) of GP 0 of Ele 0 (optimal)"] = {
      static_cast<double>(lngi_factors_.optimal_lngi_factor_lambda_2_)};
  output_data["Interpolation factor (q component 1) of GP 0 of Ele 0 (optimal)"] = {
      static_cast<double>(lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_0_)};
  output_data["Interpolation factor (q component 2) of GP 0 of Ele 0 (optimal)"] = {
      static_cast<double>(lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_1_)};
  output_data["Interpolation factor (q component 3) of GP 0 of Ele 0 (optimal)"] = {
      static_cast<double>(lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_2_)};


  output_data["LNL Residual: Interpolation factor of GP 0 of Ele 0 (optimal)"] = {
      static_cast<double>(lngi_factors_.lnl_res_optimal_lngi_factor_)};


  // write output data to csv
  csv_writer_->write_data_to_file(sim_time_, sim_timestep_, output_data);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    output_error_local_newton_loop(unsigned int step_counter)
{  // add current number of steps
  num_iters_and_steps_.eval_num_of_LNL_steps_ += step_counter;

  // stop (already started!) RMA timer
  time_measurements_.eval_time_rma_ += timers_.eval_teuchos_timer_rma_.stop();

  // output routine
  update_total();
  write_to_csv();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonData::LocalNewtonData(
    const double res_tol, const double incr_tol, const LocalNewtonConvCheck conv_check,
    const LocalNewtonDiverCont diver_cont)
    : res_tol_(res_tol), incr_tol_(incr_tol), conv_check_(conv_check), diver_cont_(diver_cont)
{
  // set number of Gauss points to 1 temporarily, since we don't
  // know it at this point in time
  num_iter_curr_timestep_.resize(1, 0);

  // reset the values (set initial 0-values to all arrays above)
  reset_all_iteration_data(0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonData::set_num_of_gp(
    const unsigned int num_of_gp)
{
  num_iter_curr_timestep_.resize(num_of_gp, num_iter_curr_timestep_[0]);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonData::
    reset_all_iteration_data(const unsigned int gp)
{
  iter_ = 0;
}

FOUR_C_NAMESPACE_CLOSE
