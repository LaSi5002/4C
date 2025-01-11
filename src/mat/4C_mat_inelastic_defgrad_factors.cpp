// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.  //
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_mat_elast_couptransverselyisotropic.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_of_time.hpp"

#include <MueLu_KeepType.hpp>
#include <NOX_Utils.H>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_Time.hpp>
#include <zlib.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>


FOUR_C_NAMESPACE_OPEN

using namespace Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

namespace
{
  // declare numerical tolerance to be used in the verification of (numerically) zero plastic strain
  // increments
  constexpr double zero_plastic_strain_increment = 1.0e-14;
  // set reference plastic strain increments for LNGI: if elastic predictor has a smaller plastic
  // strain increment than this value, then it is directly used as the initial LNL guess without
  // performing the LNGI; otherwise, the plastic predictor is updated such that its plastic strain
  // increment is smaller than this value, but higher than the numerical value of the set zero
  // plastic strain increment
  constexpr double ref_plastic_strain_increment = 1.0e-6;

  // declare file-scope instance of the constant non-material tensors
  static ConstNonMatTensors const_non_mat_tensors = ConstNonMatTensors::instance();

  // read input parameter container of parent material (i.e., underlying
  // multiplicative_split_defgrad_elasthyper material)
  Core::IO::InputParameterContainer get_parameters_of_parent_material(const int mat_id,
      const std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>>& material_map)
  {
    std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>>::const_iterator m;
    // go over material map
    for (m = material_map.begin(); m != material_map.end(); ++m)
    {
      // check type of parent material
      if (m->second->type() == Core::Materials::m_multiplicative_split_defgrad_elasthyper)
      {
        std::vector<int> inel_defgrad_facids_ =
            m->second->raw_parameters().get<std::vector<int>>("INELDEFGRADFACIDS");
        // check if the inelastic defgrad factor ids of the found multiplicative split material
        // contain the id of the considered factor
        if (std::find((inel_defgrad_facids_).begin(), (inel_defgrad_facids_).end(), mat_id) !=
            (inel_defgrad_facids_).end())
        {
          return m->second->raw_parameters();
        }
      }
    }

    FOUR_C_THROW("No parent material found for inelastic defgrad factor ID {}", mat_id);
  }

  // assemble Jacobian from components (helper function:
  // InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<10, 10> assemble_jacobian_from_components(
      const Core::LinAlg::Matrix<9, 9>& J_FdF, const Core::LinAlg::Matrix<9, 1>& J_FdS,
      const Core::LinAlg::Matrix<1, 9>& J_SdF, const double J_SdS)
  {
    // declare output Jacobian
    Core::LinAlg::Matrix<10, 10> J(Core::LinAlg::Initialization::zero);

    // set its components
    J(0, 0) = J_FdF(0, 0);
    J(0, 1) = J_FdF(0, 1);
    J(0, 2) = J_FdF(0, 2);
    J(0, 3) = J_FdF(0, 3);
    J(0, 4) = J_FdF(0, 4);
    J(0, 5) = J_FdF(0, 5);
    J(0, 6) = J_FdF(0, 6);
    J(0, 7) = J_FdF(0, 7);
    J(0, 8) = J_FdF(0, 8);
    J(0, 9) = J_FdS(0, 0);

    J(1, 0) = J_FdF(1, 0);
    J(1, 1) = J_FdF(1, 1);
    J(1, 2) = J_FdF(1, 2);
    J(1, 3) = J_FdF(1, 3);
    J(1, 4) = J_FdF(1, 4);
    J(1, 5) = J_FdF(1, 5);
    J(1, 6) = J_FdF(1, 6);
    J(1, 7) = J_FdF(1, 7);
    J(1, 8) = J_FdF(1, 8);
    J(1, 9) = J_FdS(1, 0);

    J(2, 0) = J_FdF(2, 0);
    J(2, 1) = J_FdF(2, 1);
    J(2, 2) = J_FdF(2, 2);
    J(2, 3) = J_FdF(2, 3);
    J(2, 4) = J_FdF(2, 4);
    J(2, 5) = J_FdF(2, 5);
    J(2, 6) = J_FdF(2, 6);
    J(2, 7) = J_FdF(2, 7);
    J(2, 8) = J_FdF(2, 8);
    J(2, 9) = J_FdS(2, 0);

    J(3, 0) = J_FdF(3, 0);
    J(3, 1) = J_FdF(3, 1);
    J(3, 2) = J_FdF(3, 2);
    J(3, 3) = J_FdF(3, 3);
    J(3, 4) = J_FdF(3, 4);
    J(3, 5) = J_FdF(3, 5);
    J(3, 6) = J_FdF(3, 6);
    J(3, 7) = J_FdF(3, 7);
    J(3, 8) = J_FdF(3, 8);
    J(3, 9) = J_FdS(3, 0);

    J(4, 0) = J_FdF(4, 0);
    J(4, 1) = J_FdF(4, 1);
    J(4, 2) = J_FdF(4, 2);
    J(4, 3) = J_FdF(4, 3);
    J(4, 4) = J_FdF(4, 4);
    J(4, 5) = J_FdF(4, 5);
    J(4, 6) = J_FdF(4, 6);
    J(4, 7) = J_FdF(4, 7);
    J(4, 8) = J_FdF(4, 8);
    J(4, 9) = J_FdS(4, 0);

    J(5, 0) = J_FdF(5, 0);
    J(5, 1) = J_FdF(5, 1);
    J(5, 2) = J_FdF(5, 2);
    J(5, 3) = J_FdF(5, 3);
    J(5, 4) = J_FdF(5, 4);
    J(5, 5) = J_FdF(5, 5);
    J(5, 6) = J_FdF(5, 6);
    J(5, 7) = J_FdF(5, 7);
    J(5, 8) = J_FdF(5, 8);
    J(5, 9) = J_FdS(5, 0);

    J(6, 0) = J_FdF(6, 0);
    J(6, 1) = J_FdF(6, 1);
    J(6, 2) = J_FdF(6, 2);
    J(6, 3) = J_FdF(6, 3);
    J(6, 4) = J_FdF(6, 4);
    J(6, 5) = J_FdF(6, 5);
    J(6, 6) = J_FdF(6, 6);
    J(6, 7) = J_FdF(6, 7);
    J(6, 8) = J_FdF(6, 8);
    J(6, 9) = J_FdS(6, 0);

    J(7, 0) = J_FdF(7, 0);
    J(7, 1) = J_FdF(7, 1);
    J(7, 2) = J_FdF(7, 2);
    J(7, 3) = J_FdF(7, 3);
    J(7, 4) = J_FdF(7, 4);
    J(7, 5) = J_FdF(7, 5);
    J(7, 6) = J_FdF(7, 6);
    J(7, 7) = J_FdF(7, 7);
    J(7, 8) = J_FdF(7, 8);
    J(7, 9) = J_FdS(7, 0);

    J(8, 0) = J_FdF(8, 0);
    J(8, 1) = J_FdF(8, 1);
    J(8, 2) = J_FdF(8, 2);
    J(8, 3) = J_FdF(8, 3);
    J(8, 4) = J_FdF(8, 4);
    J(8, 5) = J_FdF(8, 5);
    J(8, 6) = J_FdF(8, 6);
    J(8, 7) = J_FdF(8, 7);
    J(8, 8) = J_FdF(8, 8);
    J(8, 9) = J_FdS(8, 0);

    J(9, 0) = J_SdF(0, 0);
    J(9, 1) = J_SdF(0, 1);
    J(9, 2) = J_SdF(0, 2);
    J(9, 3) = J_SdF(0, 3);
    J(9, 4) = J_SdF(0, 4);
    J(9, 5) = J_SdF(0, 5);
    J(9, 6) = J_SdF(0, 6);
    J(9, 7) = J_SdF(0, 7);
    J(9, 8) = J_SdF(0, 8);
    J(9, 9) = J_SdS;

    return J;
  }

  // assemble additional Cmat RHS (helper function: InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<10, 6> assemble_rhs_additional_cmat(
      const Core::LinAlg::Matrix<9, 6>& min_dResFdCV,
      const Core::LinAlg::Matrix<1, 6>& min_dResSdCV)
  {
    // declare output matrix
    Core::LinAlg::Matrix<10, 6> B(Core::LinAlg::Initialization::zero);

    // set its components
    B(0, 0) = min_dResFdCV(0, 0);
    B(0, 1) = min_dResFdCV(0, 1);
    B(0, 2) = min_dResFdCV(0, 2);
    B(0, 3) = min_dResFdCV(0, 3);
    B(0, 4) = min_dResFdCV(0, 4);
    B(0, 5) = min_dResFdCV(0, 5);

    B(1, 0) = min_dResFdCV(1, 0);
    B(1, 1) = min_dResFdCV(1, 1);
    B(1, 2) = min_dResFdCV(1, 2);
    B(1, 3) = min_dResFdCV(1, 3);
    B(1, 4) = min_dResFdCV(1, 4);
    B(1, 5) = min_dResFdCV(1, 5);

    B(2, 0) = min_dResFdCV(2, 0);
    B(2, 1) = min_dResFdCV(2, 1);
    B(2, 2) = min_dResFdCV(2, 2);
    B(2, 3) = min_dResFdCV(2, 3);
    B(2, 4) = min_dResFdCV(2, 4);
    B(2, 5) = min_dResFdCV(2, 5);

    B(3, 0) = min_dResFdCV(3, 0);
    B(3, 1) = min_dResFdCV(3, 1);
    B(3, 2) = min_dResFdCV(3, 2);
    B(3, 3) = min_dResFdCV(3, 3);
    B(3, 4) = min_dResFdCV(3, 4);
    B(3, 5) = min_dResFdCV(3, 5);

    B(4, 0) = min_dResFdCV(4, 0);
    B(4, 1) = min_dResFdCV(4, 1);
    B(4, 2) = min_dResFdCV(4, 2);
    B(4, 3) = min_dResFdCV(4, 3);
    B(4, 4) = min_dResFdCV(4, 4);
    B(4, 5) = min_dResFdCV(4, 5);

    B(5, 0) = min_dResFdCV(5, 0);
    B(5, 1) = min_dResFdCV(5, 1);
    B(5, 2) = min_dResFdCV(5, 2);
    B(5, 3) = min_dResFdCV(5, 3);
    B(5, 4) = min_dResFdCV(5, 4);
    B(5, 5) = min_dResFdCV(5, 5);

    B(6, 0) = min_dResFdCV(6, 0);
    B(6, 1) = min_dResFdCV(6, 1);
    B(6, 2) = min_dResFdCV(6, 2);
    B(6, 3) = min_dResFdCV(6, 3);
    B(6, 4) = min_dResFdCV(6, 4);
    B(6, 5) = min_dResFdCV(6, 5);

    B(7, 0) = min_dResFdCV(7, 0);
    B(7, 1) = min_dResFdCV(7, 1);
    B(7, 2) = min_dResFdCV(7, 2);
    B(7, 3) = min_dResFdCV(7, 3);
    B(7, 4) = min_dResFdCV(7, 4);
    B(7, 5) = min_dResFdCV(7, 5);

    B(8, 0) = min_dResFdCV(8, 0);
    B(8, 1) = min_dResFdCV(8, 1);
    B(8, 2) = min_dResFdCV(8, 2);
    B(8, 3) = min_dResFdCV(8, 3);
    B(8, 4) = min_dResFdCV(8, 4);
    B(8, 5) = min_dResFdCV(8, 5);

    B(9, 0) = min_dResSdCV(0, 0);
    B(9, 1) = min_dResSdCV(0, 1);
    B(9, 2) = min_dResSdCV(0, 2);
    B(9, 3) = min_dResSdCV(0, 3);
    B(9, 4) = min_dResSdCV(0, 4);
    B(9, 5) = min_dResSdCV(0, 5);

    return B;
  }

  // extract the derivative of the inverse inelastic defgrad w.r.t. right CG tensor from the
  // solution of the linear system of equations. This SoE is used in the additional cmat
  // calculation. (helper function: InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<9, 6> extract_derivative_of_inv_inelastic_defgrad(
      const Core::LinAlg::Matrix<10, 6>& SOL)
  {
    // declare output derivative
    Core::LinAlg::Matrix<9, 6> diFin_dC_V(Core::LinAlg::Initialization::zero);

    // set its components
    diFin_dC_V(0, 0) = SOL(0, 0);
    diFin_dC_V(0, 1) = SOL(0, 1);
    diFin_dC_V(0, 2) = SOL(0, 2);
    diFin_dC_V(0, 3) = SOL(0, 3);
    diFin_dC_V(0, 4) = SOL(0, 4);
    diFin_dC_V(0, 5) = SOL(0, 5);

    diFin_dC_V(1, 0) = SOL(1, 0);
    diFin_dC_V(1, 1) = SOL(1, 1);
    diFin_dC_V(1, 2) = SOL(1, 2);
    diFin_dC_V(1, 3) = SOL(1, 3);
    diFin_dC_V(1, 4) = SOL(1, 4);
    diFin_dC_V(1, 5) = SOL(1, 5);

    diFin_dC_V(2, 0) = SOL(2, 0);
    diFin_dC_V(2, 1) = SOL(2, 1);
    diFin_dC_V(2, 2) = SOL(2, 2);
    diFin_dC_V(2, 3) = SOL(2, 3);
    diFin_dC_V(2, 4) = SOL(2, 4);
    diFin_dC_V(2, 5) = SOL(2, 5);

    diFin_dC_V(3, 0) = SOL(3, 0);
    diFin_dC_V(3, 1) = SOL(3, 1);
    diFin_dC_V(3, 2) = SOL(3, 2);
    diFin_dC_V(3, 3) = SOL(3, 3);
    diFin_dC_V(3, 4) = SOL(3, 4);
    diFin_dC_V(3, 5) = SOL(3, 5);

    diFin_dC_V(4, 0) = SOL(4, 0);
    diFin_dC_V(4, 1) = SOL(4, 1);
    diFin_dC_V(4, 2) = SOL(4, 2);
    diFin_dC_V(4, 3) = SOL(4, 3);
    diFin_dC_V(4, 4) = SOL(4, 4);
    diFin_dC_V(4, 5) = SOL(4, 5);

    diFin_dC_V(5, 0) = SOL(5, 0);
    diFin_dC_V(5, 1) = SOL(5, 1);
    diFin_dC_V(5, 2) = SOL(5, 2);
    diFin_dC_V(5, 3) = SOL(5, 3);
    diFin_dC_V(5, 4) = SOL(5, 4);
    diFin_dC_V(5, 5) = SOL(5, 5);

    diFin_dC_V(6, 0) = SOL(6, 0);
    diFin_dC_V(6, 1) = SOL(6, 1);
    diFin_dC_V(6, 2) = SOL(6, 2);
    diFin_dC_V(6, 3) = SOL(6, 3);
    diFin_dC_V(6, 4) = SOL(6, 4);
    diFin_dC_V(6, 5) = SOL(6, 5);

    diFin_dC_V(7, 0) = SOL(7, 0);
    diFin_dC_V(7, 1) = SOL(7, 1);
    diFin_dC_V(7, 2) = SOL(7, 2);
    diFin_dC_V(7, 3) = SOL(7, 3);
    diFin_dC_V(7, 4) = SOL(7, 4);
    diFin_dC_V(7, 5) = SOL(7, 5);

    diFin_dC_V(8, 0) = SOL(8, 0);
    diFin_dC_V(8, 1) = SOL(8, 1);
    diFin_dC_V(8, 2) = SOL(8, 2);
    diFin_dC_V(8, 3) = SOL(8, 3);
    diFin_dC_V(8, 4) = SOL(8, 4);
    diFin_dC_V(8, 5) = SOL(8, 5);

    return diFin_dC_V;
  }

  // wrap inverse inelastic defgrad and plastic strain to a vector of unknowns for the Local
  // Newton Loop (helper function: InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<10, 1> wrap_unknowns(
      const Core::LinAlg::Matrix<3, 3>& iFinM, const double& plastic_strain)
  {
    Core::LinAlg::Matrix<10, 1> x(Core::LinAlg::Initialization::zero);
    x(0) = iFinM(0, 0);
    x(1) = iFinM(1, 1);
    x(2) = iFinM(2, 2);
    x(3) = iFinM(0, 1);
    x(4) = iFinM(1, 2);
    x(5) = iFinM(0, 2);
    x(6) = iFinM(1, 0);
    x(7) = iFinM(2, 1);
    x(8) = iFinM(2, 0);
    x(9) = plastic_strain;

    return x;
  }

  // extract the inverse inelastic defgrad from the vector of unknowns used in the Local Newton
  // Loop (helper function: InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<3, 3> extract_inverse_inelastic_defgrad(const Core::LinAlg::Matrix<10, 1>& x)
  {
    Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
    iFinM(0, 0) = x(0);
    iFinM(1, 1) = x(1);
    iFinM(2, 2) = x(2);
    iFinM(0, 1) = x(3);
    iFinM(1, 2) = x(4);
    iFinM(0, 2) = x(5);
    iFinM(1, 0) = x(6);
    iFinM(2, 1) = x(7);
    iFinM(2, 0) = x(8);


    return iFinM;
  }


  // Initialize the second-order tensor interpolator for
  // InelasticDefgradTransvIsotropElastViscoplast Currently, we consider R - LOG interpolation with
  // a set exponential decay factor.
  Core::LinAlg::SecondOrderTensorInterpolator<1> init_tensor_interpolator()
  {
    // initialize interpolation parameter list
    Core::LinAlg::ScalarInterpolationParams interp_param_list;
    /// add exponential decay factor for weighting
    interp_param_list.exponential_decay_c = 20.0;

    // return corresponding tensor interpolator
    return Core::LinAlg::SecondOrderTensorInterpolator<1>{1,
        Core::LinAlg::RotationInterpolationType::RotationVector,
        Core::LinAlg::EigenvalInterpolationType::LOG, interp_param_list};
  }



  //! instance of general utilities used for analyzing the time
  //! integration algorithm within InelasticDefgradTransvIsotropElastViscoplast
  static Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils
      general_local_timint_analysis_utils;


  //! compute the inverse inelastic deformation gradient corresponding
  //! to the almost plastic predictor in InelasticDefgradTransvIsotropElastViscoplast
  //! (Note the inverse eigenvalues last_FeM_matstretch_inverse_eigenval must be
  //! ordered from small to high)
  Core::LinAlg::Matrix<3, 3> compute_inverse_plastic_defgrad_plastic_pred(
      const Core::LinAlg::Matrix<3, 3>& FM, const Core::LinAlg::Matrix<3, 3>& last_iFinM,
      const std::array<double, 3>& last_FeM_matstretch_eigenval,
      const LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvalType
          plast_pred_elastic_stretch_eigenval_type,
      const LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvectRotType
          plast_pred_elastic_stretch_eigenvect_rot_type,
      const LocalNewtonGuessInterpolation::PlasticPredictorRotationType plast_pred_rot_type)
  {
    // declare output
    Core::LinAlg::Matrix<3, 3> inv_plastic_defgrad_plastic_pred(Core::LinAlg::Initialization::zero);

    // get determinant of the current deformation gradient
    const double detF = FM.determinant();


    // compute elastic deformation gradient $ \boldsymbol{F}_{\mathrm{e}, n+1}^{0\mathrm{e}}  =
    // \boldsymbol{F}_{n+1} \boldsymbol{F}^{\text{p}^{-1}}}_n$ within the elastic predictor
    Core::LinAlg::Matrix<3, 3> Fenp_elast_pred{Core::LinAlg::Initialization::zero};
    Fenp_elast_pred.multiply_nn(1.0, FM, last_iFinM, 0.0);


    // perform polar-spectral decomposition of the elastic deformation gradient
    // within the elastic predictor
    Core::LinAlg::Matrix<3, 3> U_Fenp_elast_pred{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> lambda_Fenp_elast_pred{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> R_Fenp_elast_pred{Core::LinAlg::Initialization::zero};
    std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_Fenp_elast_pred;
    Core::LinAlg::matrix_3x3_polar_decomposition(Fenp_elast_pred, R_Fenp_elast_pred,
        U_Fenp_elast_pred, lambda_Fenp_elast_pred, spectral_pairs_Fenp_elast_pred);


    // compute elastic deformation gradient within the plastic predictor
    Core::LinAlg::Matrix<3, 3> eigenvalue_matrix{Core::LinAlg::Initialization::zero};
    switch (plast_pred_elastic_stretch_eigenval_type)
    {
      case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
          LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvalType::eliminate:
      {
        for (int i = 0; i < 3; ++i) eigenvalue_matrix(i, i) = 1.0;
        break;
      }
      case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
          LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvalType::maintain:
      {
        for (int i = 0; i < 3; ++i) eigenvalue_matrix(i, i) = last_FeM_matstretch_eigenval[i];
        break;
      }

      default:
      {
        FOUR_C_THROW("You should not be here, incompatible elastic stretch eigenvalue type {}",
            EnumTools::enum_name(plast_pred_elastic_stretch_eigenval_type));
      }
    }
    const double inv_det_eigenvalue_matrix = 1.0 / eigenvalue_matrix.determinant();
    const double det_fac = std::pow(detF * inv_det_eigenvalue_matrix, 1.0 / 3.0);
    eigenvalue_matrix.scale(det_fac);  // scale eigenvalue matrix such that the elastic deformation
                                       // gradient carries the entire determinant
    Core::LinAlg::Matrix<3, 3> eigenvector_matrix{Core::LinAlg::Initialization::zero};
    switch (plast_pred_elastic_stretch_eigenvect_rot_type)
    {
      case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
          LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvectRotType::
              elastic_predictor:
      {
        for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
          {
            eigenvector_matrix(i, j) = spectral_pairs_Fenp_elast_pred[i].second(j);
          }
        }
        break;
      }
      default:
      {
        FOUR_C_THROW(
            "You should not be here, incompatible elastic stretch eigenvector rotation type {}",
            EnumTools::enum_name(plast_pred_elastic_stretch_eigenvect_rot_type));
      }
    }
    Core::LinAlg::Matrix<3, 3> rotation_matrix{Core::LinAlg::Initialization::zero};
    switch (plast_pred_rot_type)
    {
      case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
          LocalNewtonGuessInterpolation::PlasticPredictorRotationType::elastic_predictor:
      {
        rotation_matrix = R_Fenp_elast_pred;
        break;
      }
      default:
      {
        FOUR_C_THROW(
            "You should not be here, incompatible elastic stretch eigenvector rotation type {}",
            EnumTools::enum_name(plast_pred_elastic_stretch_eigenvect_rot_type));
      }
    }
    Core::LinAlg::Matrix<3, 3> QTLambda{Core::LinAlg::Initialization::zero};
    QTLambda.multiply_tn(1.0, eigenvector_matrix, eigenvalue_matrix, 0.0);
    Core::LinAlg::Matrix<3, 3> QTLambdaQ{Core::LinAlg::Initialization::zero};
    QTLambdaQ.multiply_nn(1.0, QTLambda, eigenvector_matrix, 0.0);
    Core::LinAlg::Matrix<3, 3> elast_defgrad{Core::LinAlg::Initialization::zero};
    elast_defgrad.multiply_nn(1.0, rotation_matrix, QTLambdaQ, 0.0);
    // compute inverse deformation gradient
    Core::LinAlg::Matrix<3, 3> inv_FM{Core::LinAlg::Initialization::zero};
    inv_FM.invert(FM);

    // compute inverse plastic deformation gradient
    inv_plastic_defgrad_plastic_pred.multiply_nn(1.0, inv_FM, elast_defgrad, 0.0);

    return inv_plastic_defgrad_plastic_pred;
  }

  // precondition matrix for spectral-polar decomposition: set entries
  // smaller than a set numerical tolerance to 0
  Core::LinAlg::Matrix<3, 3> precondition_matrix(
      const Core::LinAlg::Matrix<3, 3>& input_matrix, const double num_tolerance)
  {
    Core::LinAlg::Matrix<3, 3> output_matrix{input_matrix};
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        if (std::abs(output_matrix(i, j)) < num_tolerance) output_matrix(i, j) = 0.0;
      }
    }

    return output_matrix;
  }

  // LNGI in InelasticDefgradTransvIsotropElastViscoplast: check whether two interpolation points
  // p1, p2 are equal (within a numerical tolerance)
  bool is_equal_interp_points(const LocalNewtonGuessInterpolation::InterpolationPoint& p1,
      const LocalNewtonGuessInterpolation::InterpolationPoint& p2)
  {
    // set numerical tolerance for equality
    const double num_tolerance = 1.0e-8;

    // get difference of the two interpolation points
    const double diff_p = LocalNewtonGuessInterpolation::get_diff_interp_points(p1, p2);


    return (diff_p <= num_tolerance);
  }

  // compute matrix from its spectral-polar decomposed components \f$ \boldsymbol{T}
  // = \boldsymbol{R} \boldsymbol{Q}^{T} \boldsymbol{\lambda} \boldsymbol{Q} \f$: sorted
  // eigenvalues, relative eigenvector rotation (specified using a
  // rotation vector) with respect to a reference
  // eigenvector rotation matrix, and a rotation matrix \f$ \boldsymbol{R} \f$
  Core::LinAlg::Matrix<3, 3> compute_matrix_from_decomposed_components(const double lambda_1,
      const double lambda_2, const double lambda_3,
      const Core::LinAlg::Matrix<3, 3>& reference_eigenvect_rot_matrix,
      const Core::LinAlg::Matrix<3, 1>& rel_eigenvect_rot_vect,
      const Core::LinAlg::Matrix<3, 3>& rot_matrix)
  {
    // declare output
    Core::LinAlg::Matrix<3, 3> output{Core::LinAlg::Initialization::zero};

    // construct eigenvalue matrix
    Core::LinAlg::Matrix<3, 3> eigenval_matrix{Core::LinAlg::Initialization::zero};
    eigenval_matrix(0, 0) = lambda_1;
    eigenval_matrix(1, 1) = lambda_2;
    eigenval_matrix(2, 2) = lambda_3;


    // construct eigenvector matrix
    Core::LinAlg::Matrix<3, 3> rel_eigenvect_rot_matrix =
        Core::LinAlg::calc_rot_matrix_from_rot_vect(rel_eigenvect_rot_vect);
    Core::LinAlg::Matrix<3, 3> eigenvect_rot_matrix{Core::LinAlg::Initialization::zero};
    eigenvect_rot_matrix.multiply_nn(
        1.0, reference_eigenvect_rot_matrix, rel_eigenvect_rot_matrix, 0.0);


    // construct output matrix
    Core::LinAlg::Matrix<3, 3> LQ{Core::LinAlg::Initialization::zero};
    LQ.multiply_nn(1.0, eigenval_matrix, eigenvect_rot_matrix, 0.0);
    Core::LinAlg::Matrix<3, 3> QTLQ{Core::LinAlg::Initialization::zero};
    QTLQ.multiply_tn(1.0, eigenvect_rot_matrix, LQ, 0.0);
    output.multiply_nn(1.0, rot_matrix, QTLQ, 0.0);


    return output;
  }

  using Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::debug_mode;


}  // namespace


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradNoGrowth::InelasticDefgradNoGrowth(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
  // do nothing here
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradScalar::InelasticDefgradScalar(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      scalar1_(matdata.parameters.get<int>("SCALAR1")),
      scalar1_ref_conc_(matdata.parameters.get<double>("SCALAR1_RefConc"))
{
  // safety checks
  // in case not all scatra dofs are transported scalars, the last scatra dof is a potential
  // and can not be treated as a concentration but it is treated like that in
  // so3_scatra_evaluate.cpp in the pre_evaluate method!
  if (scalar1_ != 1) FOUR_C_THROW("At the moment it is only possible that SCALAR1 induces growth");
  if (matdata.parameters.get<double>("SCALAR1_RefConc") < 0.0)
    FOUR_C_THROW("The reference concentration of SCALAR1 can't be negative");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradLinScalar::InelasticDefgradLinScalar(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradScalar(matdata),
      scalar1_molar_growth_fac_(matdata.parameters.get<double>("SCALAR1_MolarGrowthFac"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradIntercalFrac::InelasticDefgradIntercalFrac(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradScalar(matdata)
{
  // get matid
  const int matid = matdata.parameters.get<int>("MATID");

  // Check if the material specified by user with MATID is an electrode material
  if (matid > 0)
  {
    // retrieve problem instance to read from
    const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
    // retrieve validated input line of material ID in question
    auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
    switch (curmat->type())
    {
      case Core::Materials::m_electrode:
      {
        // Get C_max and Chi_max of electrode material
        c_max_ = curmat->raw_parameters().get<double>("C_MAX");
        chi_max_ = curmat->raw_parameters().get<double>("CHI_MAX");
        break;
      }
      default:
        FOUR_C_THROW("The material you have specified by MATID has to be an electrode material!");
    }
  }
  else
  {
    FOUR_C_THROW("You have to enter a valid MATID for the corresponding electrode material!");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradPolyIntercalFrac::InelasticDefgradPolyIntercalFrac(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradIntercalFrac(matdata),
      poly_coeffs_(matdata.parameters.get<std::vector<double>>("POLY_PARAMS")),
      x_max_(matdata.parameters.get<double>("X_max")),
      x_min_(matdata.parameters.get<double>("X_min"))
{
  // safety check
  if (poly_coeffs_.size() !=
      static_cast<unsigned int>(matdata.parameters.get<int>("POLY_PARA_NUM")))
  {
    FOUR_C_THROW(
        "Number of coefficients POLY_PARA_NUM you entered in input file has to match the "
        "size "
        "of coefficient vector POLY_PARAMS");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradLinScalarAniso::InelasticDefgradLinScalarAniso(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradLinScalar(matdata),
      growth_dir_(std::make_shared<InelasticDeformationDirection>(
          matdata.parameters.get<std::vector<double>>("GrowthDirection")))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradPolyIntercalFracAniso::InelasticDefgradPolyIntercalFracAniso(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradPolyIntercalFrac(matdata),
      growth_dir_(std::make_shared<InelasticDeformationDirection>(
          matdata.parameters.get<std::vector<double>>("GrowthDirection")))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDeformationDirection::InelasticDeformationDirection(
    std::vector<double> growthdirection)
    : growth_dir_mat_()
{
  if (growthdirection.size() != 3)
  {
    FOUR_C_THROW(
        "Since we have a 3D problem here, vector that defines the growth direction also "
        "needs "
        "to "
        "have the size 3!");
  }

  // fill matrix that determines the growth direction
  const double growthdirvecnorm =
      std::sqrt(std::pow(growthdirection[0], 2.0) + std::pow(growthdirection[1], 2.0) +
                std::pow(growthdirection[2], 2.0));
  const double invquadrgrowthdirvecnorm = 1.0 / (growthdirvecnorm * growthdirvecnorm);

  // loop over all rows and columns to fill the matrix and scale it correctly on the fly
  for (unsigned i = 0; i < growthdirection.size(); ++i)
  {
    for (unsigned j = 0; j < growthdirection.size(); ++j)
    {
      growth_dir_mat_(i, j) = invquadrgrowthdirvecnorm * growthdirection[i] * growthdirection[j];
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradLinTempIso::InelasticDefgradLinTempIso(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      ref_temp_(matdata.parameters.get<double>("RefTemp")),
      temp_growth_fac_(matdata.parameters.get<double>("Temp_GrowthFac"))
{
  // safety checks
  if (ref_temp_ < 0.0) FOUR_C_THROW("Avoid negative reference temperatures");
  if (temp_growth_fac_ == 0.0)
  {
    FOUR_C_THROW(
        "Do not use 'MAT_InelasticDefgradLinTempIso' with a growth factor of 0.0. Use "
        "'MAT_InelasticDefgradNoGrowth' instead!");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradTimeFunct::InelasticDefgradTimeFunct(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), funct_num_(matdata.parameters.get<int>("FUNCT_NUM"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast::
    InelasticDefgradTransvIsotropElastViscoplast(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      viscoplastic_law_id_(matdata.parameters.get<int>("VISCOPLAST_LAW_ID")),
      fiber_reader_gid_(matdata.parameters.get<int>("FIBER_READER_ID")),
      yield_cond_a_(matdata.parameters.get<double>("YIELD_COND_A")),
      yield_cond_b_(matdata.parameters.get<double>("YIELD_COND_B")),
      yield_cond_f_(matdata.parameters.get<double>("YIELD_COND_F")),
      mat_behavior_(matdata.parameters.get<MatBehavior>("MAT_BEHAVIOR")),
      timint_type_(matdata.parameters.get<TimIntType>("TIME_INTEGRATION_HIST_VARS")),
      linearization_type_(matdata.parameters.get<LinearizationType>("LINEARIZATION")),
      max_plastic_strain_incr_(matdata.parameters.get<double>("MAX_PLASTIC_STRAIN_INCR")),
      max_plastic_strain_deriv_incr_(
          matdata.parameters.get<double>("MAX_PLASTIC_STRAIN_DERIV_INCR")),
      use_lngi_(matdata.parameters.get<bool>("USE_LNGI")),
      lngi_plastic_pred_elastic_stretch_eigenval_type_(matdata.parameters
              .get<LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvalType>(
                  "LNGI_PLASTIC_PRED_ELASTIC_STRETCH_EIGENVAL_TYPE")),
      lngi_plastic_pred_elastic_stretch_eigenvect_rot_type_(matdata.parameters
              .get<LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvectRotType>(
                  "LNGI_PLASTIC_PRED_ELASTIC_STRETCH_EIGENVECT_ROT_TYPE")),
      lngi_plastic_pred_rot_type_(
          matdata.parameters.get<LocalNewtonGuessInterpolation::PlasticPredictorRotationType>(
              "LNGI_PLASTIC_PRED_ROT_TYPE")),
      lngi_starting_point_type_(matdata.parameters
              .get<LocalNewtonGuessInterpolation::LocalNewtonGuessInterpolationStartingPointType>(
                  "LNGI_STARTING_POINT_TYPE")),
      lngi_starting_point_(matdata.parameters.get<double>("LNGI_STARTING_POINT")),
      lngi_interval_scan_param_(matdata.parameters.get<double>("LNGI_INTERVAL_SCAN_PARAM")),
      lngi_max_num_reinterp_(matdata.parameters.get<int>("LNGI_MAX_NUM_REINTERP")),
      lngi_min_interp_interval_(matdata.parameters.get<double>("LNGI_MIN_INTERP_INTERVAL")),
      lngi_reinterp_min_rel_dev_(matdata.parameters.get<double>("LNGI_REINTERP_MIN_REL_DEV")),
      lngi_precondition_matrices_(matdata.parameters.get<bool>("LNGI_PRECONDITION_MATRICES")),
      lngi_precondition_matrices_num_tol_(
          matdata.parameters.get<double>("LNGI_PRECONDITION_MATRICES_NUM_TOL")),
      lngi_check_consistency_(matdata.parameters.get<bool>("LNGI_CHECK_CONSISTENCY")),
      use_steepest_descent_update_correction_(
          matdata.parameters.get<bool>("USE_STEEPEST_DESCENT_UPDATE_CORRECTION")),
      use_line_search_(matdata.parameters.get<bool>("USE_LINE_SEARCH")),
      check_line_search_angle_condition_(
          matdata.parameters.get<bool>("CHECK_LINE_SEARCH_ANGLE_CONDITION")),
      line_search_angle_condition_tolerance_(
          matdata.parameters.get<double>("LINE_SEARCH_ANGLE_CONDITION_TOLERANCE")),
      use_substepping_(matdata.parameters.get<bool>("USE_SUBSTEPPING")),
      analyze_timint_(matdata.parameters.get<bool>("ANALYZE_TIMINT")),
      analyze_timint_timer_rel_tol_(matdata.parameters.get<double>("ANALYZE_TIMINT_TIMER_REL_TOL")),
      max_substepping_halve_num_(matdata.parameters.get<int>("MAX_SUBSTEPPING_HALVE_NUM")),
      mat_exp_calc_method_(
          matdata.parameters.get<Core::LinAlg::MatrixExpCalcMethod>("MATRIX_EXP_CALC_METHOD")),
      mat_exp_deriv_calc_method_(
          matdata.parameters.get<Core::LinAlg::GenMatrixExpFirstDerivCalcMethod>(
              "MATRIX_EXP_DERIV_CALC_METHOD")),
      mat_log_calc_method_(
          matdata.parameters.get<Core::LinAlg::MatrixLogCalcMethod>("MATRIX_LOG_CALC_METHOD")),
      mat_log_deriv_calc_method_(
          matdata.parameters.get<Core::LinAlg::GenMatrixLogFirstDerivCalcMethod>(
              "MATRIX_LOG_DERIV_CALC_METHOD")),
      local_newton_res_tol_(matdata.parameters.get<double>("LOCAL_NEWTON_RES_TOL")),
      local_newton_incr_tol_(matdata.parameters.get<double>("LOCAL_NEWTON_INCR_TOL")),
      local_newton_conv_check_(
          matdata.parameters.get<LocalNewtonConvCheck>("LOCAL_NEWTON_CONV_CHECK")),
      local_newton_diver_cont_(
          matdata.parameters.get<LocalNewtonDiverCont>("LOCAL_NEWTON_DIVER_CONT")),
      use_csv_output_failed_local_newton_iter_(
          matdata.parameters.get<bool>("USE_CSV_OUTPUT_FAILED_LOCAL_NEWTON_ITER")),
      use_csv_output_lngi_micro_iter_(
          matdata.parameters.get<bool>("USE_CSV_OUTPUT_LNGI_MICRO_ITER")),
      use_csv_output_line_search_micro_iter_(
          matdata.parameters.get<bool>("USE_CSV_OUTPUT_LINE_SEARCH_MICRO_ITER"))
{
  // consistency checks
  if (max_substepping_halve_num_ < 0) FOUR_C_THROW("Parameter MAX_HALVE_NUM_SUBSTEP must be >= 0!");
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradFactors::InelasticDefgradFactors(Core::Mat::PAR::Parameter* params)
    : params_(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::shared_ptr<Mat::InelasticDefgradFactors> Mat::InelasticDefgradFactors::factory(int matnum)
{
  // for the sake of safety
  if (Global::Problem::instance()->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");

  // another safety check
  if (Global::Problem::instance()->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // check correct masslin type
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  if (Teuchos::getIntegralValue<Inpar::Solid::MassLin>(sdyn, "MASSLIN") !=
      Inpar::Solid::MassLin::ml_none)
  {
    FOUR_C_THROW(
        "If you use the material 'InelasticDefgradFactors' please set 'MASSLIN' in the "
        "STRUCTURAL DYNAMIC Section to 'None', or feel free to implement other "
        "possibility!");
  }

  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matnum);

  // get material type and call corresponding constructors
  const Core::Materials::MaterialType currentMaterialType = curmat->type();
  switch (currentMaterialType)
  {
    case Core::Materials::mfi_no_growth:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradNoGrowth*>(curmat);

      return std::make_shared<InelasticDefgradNoGrowth>(params);
    }
    case Core::Materials::mfi_lin_scalar_aniso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradLinScalarAniso*>(curmat);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradLinScalarAniso>(params);
    }
    case Core::Materials::mfi_lin_scalar_iso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradScalar*>(curmat);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradLinScalarIso>(params);
    }
    case Core::Materials::mfi_poly_intercal_frac_aniso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFracAniso*>(curmat);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradPolyIntercalFracAniso>(params);
    }
    case Core::Materials::mfi_poly_intercal_frac_iso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFrac*>(curmat);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradPolyIntercalFracIso>(params);
    }

    case Core::Materials::mfi_lin_temp_iso:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradLinTempIso*>(curmat);
      return std::make_shared<InelasticDefgradLinTempIso>(params);
    }
    case Core::Materials::mfi_time_funct:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradTimeFunct*>(curmat);
      return std::make_shared<InelasticDefgradTimeFunct>(params);
    }
    case Core::Materials::mfi_transv_isotrop_elast_viscoplast:
    {
      // read material map of the global problem
      std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>> material_map =
          Global::Problem::instance(probinst)->materials()->map();

      // retrieve parameter container of parent material
      Core::IO::InputParameterContainer parentmat_input_params =
          get_parameters_of_parent_material(matnum, material_map);

      // create parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast*>(curmat);

      // create viscoplastic law
      auto viscoplastic_law = Mat::Viscoplastic::Law::factory(params->viscoplastic_law_id());

      // construct fiber reader
      auto* fiber_reader_params = Global::Problem::instance(probinst)->materials()->parameter_by_id(
          params->fiber_reader_gid());
      FOUR_C_ASSERT_ALWAYS(
          fiber_reader_params->type() == Core::Materials::mes_couptransverselyisotropic,
          "Provided fiber reader material is not of the correct type (hyperelastic, "
          "transversely "
          "isotropic: ELAST_CoupTransverselyIsotropic)!");
      Mat::Elastic::CoupTransverselyIsotropic fiber_reader{
          dynamic_cast<Mat::Elastic::PAR::CoupTransverselyIsotropic*>(fiber_reader_params)};

      // retrieve elastic materials
      std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsumel;
      std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> potsumel_transviso;
      for (int matid_elastic : parentmat_input_params.get<std::vector<int>>("MATIDSEL"))
      {
        // create elastic component
        auto elastic_summand = Mat::Elastic::Summand::factory(matid_elastic);
        FOUR_C_ASSERT_ALWAYS(elastic_summand != nullptr, "Failed to allocate");
        // add to the list of elastic components
        if (elastic_summand->material_type() == Core::Materials::mes_couptransverselyisotropic)
        {
          potsumel_transviso.push_back(
              std::dynamic_pointer_cast<Mat::Elastic::CoupTransverselyIsotropic>(elastic_summand));
        }
        else
        {
          potsumel.push_back(elastic_summand);
        }
      }

      // return shared pointer to the inelastic factor
      return std::make_shared<InelasticDefgradTransvIsotropElastViscoplast>(
          params, viscoplastic_law, fiber_reader, potsumel, potsumel_transviso);
    }

    default:
      FOUR_C_THROW("cannot deal with type {}", curmat->type());
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradScalar::InelasticDefgradScalar(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params), concentrations_(nullptr)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradScalar::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp, const int eleGID)
{
  // store scalars of current gauss point
  concentrations_ = params.get<std::shared_ptr<std::vector<double>>>("scalars");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradScalar::set_concentration_gp(const double concentration)
{
  const int scalar1 = parameter()->scalar1();
  concentrations_->at(scalar1 - 1) = concentration;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradPolyIntercalFrac::InelasticDefgradPolyIntercalFrac(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradScalar(params)
{
  polynomial_growth_ = std::make_shared<InelasticDefgradPolynomialShape>(
      parameter()->poly_coeffs(), parameter()->x_min(), parameter()->x_max());

  // get reference intercalation fraction
  const double x_ref = Mat::Electrode::compute_intercalation_fraction(
      parameter()->scalar1_ref_conc(), parameter()->chimax(), parameter()->cmax(), 1.0);

  // set the polynomial value in the reference configuration
  parameter()->set_polynom_reference_value(polynomial_growth_->compute_polynomial(x_ref));
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolyIntercalFrac::evaluate_polynomial(
    const double concentration, const double detjacobian)
{
  // get intercalation fraction
  const double x = Mat::Electrode::compute_intercalation_fraction(
      concentration, parameter()->chimax(), parameter()->cmax(), detjacobian);

  // check bounds of validity of polynomial
  polynomial_growth_->check_polynomial_bounds(x);

  // calculate and return the value of the polynomial
  return polynomial_growth_->compute_polynomial(x);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolyIntercalFrac::evaluate_polynomial_derivative(
    const double concentration, const double detjacobian)
{
  // get intercalation fraction
  const double x = Mat::Electrode::compute_intercalation_fraction(
      concentration, parameter()->chimax(), parameter()->cmax(), detjacobian);

  return polynomial_growth_->compute_polynomial_derivative(x);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradPolyIntercalFrac::get_inelastic_source()
{
  return Mat::PAR::InelasticSource::concentration;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinScalarIso::InelasticDefgradLinScalarIso(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradScalar(params)
{
  linear_growth_ = std::make_shared<InelasticDefgradLinearShape>(
      parameter()->scalar1_molar_growth_fac(), parameter()->scalar1_ref_conc());
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradLinScalarIso::get_inelastic_source()
{
  return Mat::PAR::InelasticSource::concentration;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarIso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // get parameter
  const int sc1 = parameter()->scalar1();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * defgrad->determinant();

  // get growth factor
  const double growth_factor = linear_growth_->evaluate_linear_growth(material_concentration);

  const double isoinelasticdefo = std::pow(1.0 + growth_factor, (1.0 / 3.0));

  // calculate inverse inelastic deformation gradient
  for (int i = 0; i < 3; ++i) iFinM(i, i) = 1.0 / isoinelasticdefo;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarIso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  // static variables
  static Core::LinAlg::Matrix<9, 6> diFinjdC(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);

  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double sc1GrowthFac = linear_growth_->growth_fac();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();

  // get growth factor
  const double growth_factor = linear_growth_->evaluate_linear_growth(concentration * detjacobian);

  // evaluate scaling factor
  const double scalefac =
      -sc1GrowthFac * concentration * detjacobian / 6.0 * std::pow(1 + growth_factor, -4.0 / 3.0);

  // calculate diFindC
  diFinjdC.multiply_nt(scalefac, id9x1, iCV, 0.0);

  // cmatadd = 2 dSdiFinj : diFinjdC
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarIso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);
  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double sc1GrowthFac = linear_growth_->growth_fac();
  const double detjacobian = defgrad->determinant();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * detjacobian;

  // get growth factor
  const double growth_factor = linear_growth_->evaluate_linear_growth(material_concentration);

  // calculate scalefac
  const double scalefac =
      -sc1GrowthFac / 3.0 * detjacobian * std::pow(1 + growth_factor, -4.0 / 3.0);

  // calculate diFindc and add contribution to dstressdc = dSdiFinj : diFinjdc
  dstressdc.multiply_nn(scalefac, dSdiFinj, id9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarIso::evaluate_inelastic_def_grad_derivative(
    const double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  // get parameters
  const int sc1 = parameter()->scalar1();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * detjacobian;

  // get growth factor
  const double growth_factor = linear_growth_->evaluate_linear_growth(material_concentration);
  // calculate the scale factor needed to calculate the derivative below
  const double scalefac = 1.0 / 3.0 * std::pow(1 + growth_factor, -2.0 / 3.0) *
                          linear_growth_->growth_fac() * detjacobian;

  dFindx =
      Core::LinAlg::get_full(scalefac * Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinScalarAniso::InelasticDefgradLinScalarAniso(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradScalar(params)
{
  linear_growth_ = std::make_shared<InelasticDefgradLinearShape>(
      parameter()->scalar1_molar_growth_fac(), parameter()->scalar1_ref_conc());
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradLinScalarAniso::get_inelastic_source()
{
  return Mat::PAR::InelasticSource::concentration;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarAniso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // init and clear variable
  static Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  FinM.clear();
  Core::LinAlg::TensorView<double, 3, 3> Fin = Core::LinAlg::make_tensor_view(FinM);

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * defgrad->determinant();

  // get growth factor
  const double growth_factor = linear_growth_->evaluate_linear_growth(material_concentration);

  // calculate inelastic deformation gradient and its inverse
  for (int i = 0; i < 3; ++i) FinM(i, i) = 1.0;

  // finalize inelastic deformation gradient matrix (FinM is calculated, such that the
  // volume change is a linear function of the scalar (mapped to reference frame) that
  // causes it)
  Fin += growth_factor * Core::LinAlg::get_full(parameter()->growth_dir_mat());

  // calculate inverse of inelastic deformation gradient matrix
  iFinM.invert(FinM);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarAniso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  static Core::LinAlg::Matrix<3, 3> temp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinjGiFinj(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> iFinjGiFinj9x1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 6> diFinjdC(Core::LinAlg::Initialization::zero);

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double sc1GrowthFac = linear_growth_->growth_fac();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();

  // prepare scalefac
  const double scalefac = -sc1GrowthFac * concentration * detjacobian / 2.0;

  // calculate F_{in,j}^{-1} . G . F_{in,j}^{-1} with F_{in,j}, the j-th factor of F_{in}
  temp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 0.0);
  iFinjGiFinj.multiply_nn(1.0, temp, iFinjM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(iFinjGiFinj, iFinjGiFinj9x1);

  // calculate diFinjdC
  diFinjdC.multiply_nt(scalefac, iFinjGiFinj9x1, iCV, 0.0);

  // cmatadd = 2 dSdiFinj : diFinjdC
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarAniso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  // static variables
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> diFinjdcM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> diFinjdc9x1(Core::LinAlg::Initialization::zero);

  // get parameters
  const double sc1GrowthFac = linear_growth_->growth_fac();
  const double detjacobian = defgrad->determinant();

  // prepare scalefac
  const double scalefac = -sc1GrowthFac * detjacobian;

  // calculate diFinjdc
  tmp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 0.0);
  diFinjdcM.multiply_nn(scalefac, tmp, iFinjM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(diFinjdcM, diFinjdc9x1);

  // dstressdc = dSdiFinj : diFinjdc
  dstressdc.multiply_nn(1.0, dSdiFinj, diFinjdc9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarAniso::evaluate_inelastic_def_grad_derivative(
    const double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  const double scalefac = linear_growth_->growth_fac() * detjacobian;

  dFindx = Core::LinAlg::get_full(parameter()->growth_dir_mat());
  dFindx *= scalefac;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradPolyIntercalFracIso::InelasticDefgradPolyIntercalFracIso(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradPolyIntercalFrac(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracIso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // get parameters
  const int sc1 = parameter()->scalar1();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomial
  const double polynomValue =
      evaluate_polynomial(get_concentration_gp().at(sc1 - 1), defgrad->determinant());

  // calculate growth
  const double isoInelasticDefo =
      std::pow((1.0 + polynomValue) / (1.0 + polynomReferenceValue), (1.0 / 3.0));
  // calculate inverse inelastic deformation gradient
  for (int i = 0; i < 3; ++i) iFinM(i, i) = 1.0 / isoInelasticDefo;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracIso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  // static variables
  static Core::LinAlg::Matrix<9, 6> diFinjdC(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);

  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double chi_max = parameter()->chimax();
  const double c_max = parameter()->cmax();
  const double detjacobian = defgrad->determinant();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomials
  const double polynomValue = evaluate_polynomial(concentration, detjacobian);
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);

  // prepare scalefac
  const double scalefac = -1.0 / (6.0 * c_max) * concentration * chi_max * detjacobian *
                          std::pow(1.0 + polynomValue, -4.0 / 3.0) * polynomDerivativeValue *
                          std::pow(1.0 + polynomReferenceValue, 1.0 / 3.0);

  // calculate diFinjdC
  diFinjdC.multiply_nt(scalefac, id9x1, iCV, 0.0);

  // cmatadd = 2 dSdiFinj : diFinjdC
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracIso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);
  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomial and derivatives
  const double polynomValue = evaluate_polynomial(concentration, detjacobian);
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);
  const double dChidc = Mat::Electrode::compute_d_intercalation_fraction_d_concentration(
      parameter()->chimax(), parameter()->cmax(), detjacobian);

  // prepare scalefac
  const double scalefac = -1.0 / 3.0 * std::pow(1.0 + polynomValue, -4.0 / 3.0) *
                          std::pow(1.0 + polynomReferenceValue, 1.0 / 3.0) *
                          polynomDerivativeValue * dChidc;

  // calculate diFinjdc and add contribution to dstressdc = dSdiFinj : diFinjdc
  dstressdc.multiply_nn(scalefac, dSdiFinj, id9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracIso::evaluate_inelastic_def_grad_derivative(
    const double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  // get parameters
  const int sc1 = parameter()->scalar1();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomial and its derivative
  const double polynomValue = evaluate_polynomial(concentration, detjacobian);
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);

  // calculate the scale factor needed to get the derivative later
  const double denominator = 1.0 / (polynomReferenceValue + 1.0);
  const double base = (polynomValue + 1.0) * denominator;
  const double dChidc = Mat::Electrode::compute_d_intercalation_fraction_d_concentration(
      parameter()->chimax(), parameter()->cmax(), detjacobian);
  const double scalefac =
      1.0 / 3.0 * std::pow(base, -2.0 / 3.0) * polynomDerivativeValue * denominator * dChidc;

  // here dFindc is zeroed out and filled with the current value
  dFindx =
      Core::LinAlg::get_full(scalefac * Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradPolyIntercalFracAniso::InelasticDefgradPolyIntercalFracAniso(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradPolyIntercalFrac(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracAniso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // init and clear variable
  static Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  FinM.clear();

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomials
  const double polynomValue =
      evaluate_polynomial(get_concentration_gp().at(sc1 - 1), defgrad->determinant());

  // calculate growth factor
  const double growth_factor =
      (polynomValue - polynomReferenceValue) / (polynomReferenceValue + 1.0);

  // calculate inelastic deformation gradient and its inverse
  for (int i = 0; i < 3; ++i) FinM(i, i) = 1.0;

  // add the growth part
  FinM.update(growth_factor,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 1.0);

  // calculate inverse of inelastic deformation gradient matrix
  iFinM.invert(FinM);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracAniso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  static Core::LinAlg::Matrix<3, 3> temp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinjGiFinj(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> iFinjGiFinj9x1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 6> diFinjdC(Core::LinAlg::Initialization::zero);

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double chi_max = parameter()->chimax();
  const double c_max = parameter()->cmax();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get first derivative of polynomial
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);

  // prepare scalefac
  const double scalefac = -detjacobian * concentration * chi_max * polynomDerivativeValue /
                          (2.0 * c_max * (polynomReferenceValue + 1.0));

  // calculate F_{in,j}^{-1} . G . F_{in,j}^{-1} with F_{in,j}, the j-th factor of F_{in}
  temp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 0.0);
  iFinjGiFinj.multiply_nn(1.0, temp, iFinjM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(iFinjGiFinj, iFinjGiFinj9x1);

  // calculate diFinjdC
  diFinjdC.multiply_nt(scalefac, iFinjGiFinj9x1, iCV, 0.0);

  // cmatadd = 2 dSdiFinj : diFinjdC
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracAniso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  // static variables
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> diFinjdcM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> diFinjdc9x1(Core::LinAlg::Initialization::zero);

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double detjacobian = defgrad->determinant();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get derivatives
  const double polynomDerivativeValue =
      evaluate_polynomial_derivative(get_concentration_gp().at(sc1 - 1), detjacobian);
  const double dChidc = Mat::Electrode::compute_d_intercalation_fraction_d_concentration(
      parameter()->chimax(), parameter()->cmax(), detjacobian);

  // prepare scalefac
  const double scalefac = -polynomDerivativeValue / (polynomReferenceValue + 1.0) * dChidc;

  // calculate diFinjdc
  tmp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 0.0);
  diFinjdcM.multiply_nn(scalefac, tmp, iFinjM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(diFinjdcM, diFinjdc9x1);

  // dstressdc = dSdiFinj : diFinjdc
  dstressdc.multiply_nn(1.0, dSdiFinj, diFinjdc9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracAniso::evaluate_inelastic_def_grad_derivative(
    const double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  // get parameters
  const int sc1 = parameter()->scalar1();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomial derivative
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);

  const double dChidc = Mat::Electrode::compute_d_intercalation_fraction_d_concentration(
      parameter()->chimax(), parameter()->cmax(), detjacobian);
  const double scalefac = polynomDerivativeValue / (polynomReferenceValue + 1.0) * dChidc;

  // here dFindc is zeroed out and filled with the current value
  dFindx = Core::LinAlg::get_full(scalefac * parameter()->growth_dir_mat());
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinearShape::InelasticDefgradLinearShape(
    const double growth_fac, const double reference_value)
    : growth_fac_(growth_fac), reference_value_(reference_value)
{
  // safety checks
  if (growth_fac < 0.0)
    FOUR_C_THROW("Growth factor can not be negative, please check your input file!");
  if (growth_fac == 0.0)
  {
    FOUR_C_THROW(
        "Do not use linear growth laws with a growth factor of 0.0. Use "
        "'MAT_InelasticDefgradNoGrowth' instead!");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradLinearShape::evaluate_linear_growth(const double value) const
{
  // calculate and return the linear growth factor
  return growth_fac_ * (value - reference_value_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradPolynomialShape::InelasticDefgradPolynomialShape(
    std::vector<double> poly_coeffs, const double x_min, const double x_max)
    : poly_coeffs_(std::move(poly_coeffs)), x_min_(x_min), x_max_(x_max)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolynomialShape::compute_polynomial(const double x)
{
  // initialize the variable for the evaluation of the polynomial
  double polynom(0.0);

  // compute polynomial
  for (unsigned i = 0; i < poly_coeffs_.size(); ++i) polynom += poly_coeffs_[i] * std::pow(x, i);

  return polynom;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolynomialShape::compute_polynomial_derivative(const double x)
{
  // initialize the variable for the derivative of the polynomial
  double polynomDerivative(0.0);

  // compute first derivative of polynomial
  for (unsigned i = 1; i < poly_coeffs_.size(); ++i)
    polynomDerivative += i * poly_coeffs_[i] * std::pow(x, i - 1);

  return polynomDerivative;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolynomialShape::check_polynomial_bounds(const double x) const
{
  // safety check for validity of polynomial
  if ((x < x_min_) or (x > x_max_))
  {
    std::cout << "WARNING: Polynomial is evaluated outside its range of validity!" << std::endl;
    std::cout << "Evaluation at: " << x << " Lower bound is " << x_min_ << " Upper bound is "
              << x_max_ << std::endl;
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinTempIso::InelasticDefgradLinTempIso(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp, const int eleGID)
{
  temperature_ = params.get<double>("temperature");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // get parameters
  const double tempgrowthfac = parameter()->get_temp_growth_fac();
  const double reftemp = parameter()->ref_temp();

  const double growthfactor = 1.0 + tempgrowthfac * (temperature_ - reftemp);
  if (growthfactor <= 0.0) FOUR_C_THROW("Determinante of growth must not become negative");
  const double isoinelasticdefo = std::pow(growthfactor, (1.0 / 3.0));

  // calculate inverse inelastic deformation gradient
  for (int i = 0; i < 3; ++i) iFinM(i, i) = 1.0 / isoinelasticdefo;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::evaluate_inelastic_def_grad_derivative(
    double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  // get parameters
  const double tempgrowthfac = parameter()->get_temp_growth_fac();
  const double reftemp = parameter()->ref_temp();

  const double growthfactor = 1.0 + tempgrowthfac * (temperature_ - reftemp);
  const double scalefac = tempgrowthfac / 3.0 * std::pow(growthfactor, -2.0 / 3.0);

  // here dFindT is zeroed out and filled with the current value
  dFindx =
      Core::LinAlg::get_full(scalefac * Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  // nothing to do so far, as current growth model is not a function of displacements (and
  // thus C)
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdT)
{
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);
  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters from parameter class
  const double tempgrowthfac = parameter()->get_temp_growth_fac();
  const double reftemp = parameter()->ref_temp();

  const double growthfactor = 1.0 + tempgrowthfac * (temperature_ - reftemp);
  if (growthfactor <= 0.0) FOUR_C_THROW("Determinante of growth must not become negative");

  const double scalefac = -tempgrowthfac / (3.0 * std::pow(growthfactor, 4.0 / 3.0));

  // dstressdT = dSdiFinj : diFinjdT
  // diFinjdT = - growthfac/(3*[1 + growthfac*(T-T_{ref})]^(4/3)) * I
  dstressdT.multiply_nn(scalefac, dSdiFinj, id9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradLinTempIso::get_inelastic_source()
{
  return PAR::InelasticSource::temperature;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::evaluate_inelastic_def_grad_derivative(
    double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  iFinM = identity_;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
    Core::LinAlg::Matrix<6, 1>& dstressdx)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradNoGrowth::get_inelastic_source()
{
  return PAR::InelasticSource::none;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradNoGrowth::InelasticDefgradNoGrowth(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params), identity_(Core::LinAlg::Initialization::zero)
{
  // add 1.0 to main diagonal
  identity_(0, 0) = identity_(1, 1) = identity_(2, 2) = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp, const int eleGID)
{
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTimeFunct::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  const double idetFin = std::pow(funct_value_, -1.0 / 3.0);
  iFinM.update(idetFin, identity_, 0.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradTimeFunct::get_inelastic_source()
{
  return PAR::InelasticSource::none;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTimeFunct::InelasticDefgradTimeFunct(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params),
      funct_value_(0.0),
      identity_(Core::LinAlg::Initialization::zero)
{
  // add 1.0 to main diagonal
  identity_(0, 0) = identity_(1, 1) = identity_(2, 2) = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTimeFunct::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp, const int eleGID)
{
  // evaluate function value for current time step.
  auto& funct = Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
      parameter()->funct_num());
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  const double time = *context.total_time;
  funct_value_ = funct.evaluate(time);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplast::InelasticDefgradTransvIsotropElastViscoplast(
    Core::Mat::PAR::Parameter* params, std::shared_ptr<Mat::Viscoplastic::Law> viscoplastic_law,
    Mat::Elastic::CoupTransverselyIsotropic fiber_reader,
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> pot_sum_el,
    std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> pot_sum_el_transv_iso)
    : InelasticDefgradFactors(params),
      potsumel_(std::move(pot_sum_el)),
      potsumel_transviso_(std::move(pot_sum_el_transv_iso)),
      viscoplastic_law_(std::move(viscoplastic_law)),
      fiber_reader_(std::move(fiber_reader)),
      tensor_interpolator_{init_tensor_interpolator()},
      lnl_guess_interpolation_(parameter()->lngi_interval_scan_param(),
          parameter()->lngi_max_num_reinterp(),
          parameter()->lngi_plastic_pred_elastic_stretch_eigenval_type(),
          parameter()->lngi_plastic_pred_elastic_stretch_eigenvect_rot_type(),
          parameter()->lngi_plastic_pred_rot_type(), parameter()->lngi_min_interp_interval()),
      globiter_(-1),  // initialized as -1, because this is called one time even
                      // prior to the first global predictor evaluation
      csv_output_lngi_micro_iter_data_{CSVOutputTrackingData{}},
      csv_output_line_search_micro_iter_data_{CSVOutputTrackingData{}},
      lnl_data_(parameter()->local_newton_res_tol(), parameter()->local_newton_incr_tol(),
          parameter()->local_newton_conv_check(), parameter()->local_newton_diver_cont())
{
  // set time step size to 0.0 (this is set to the correct and current value in the
  // preevaluate method)
  time_step_tracker_.dt_ = 0.0;
  // set minimum substep length
  time_step_tracker_.min_dt_ = 0.0;

  // ----- set last_ and current_ variables referring to values at different time instants
  // ----- for now: the number of Gauss points is unknown -> we set the values only for 1
  // Gauss point and update the number of Gauss points in the setup method

  // default values of the inverse plastic deformation gradient: unit tensor
  time_step_quantities_.last_plastic_defgrad_inverse_.resize(1, const_non_mat_tensors.id3x3_);
  time_step_quantities_.current_plastic_defgrad_inverse_.resize(
      1, const_non_mat_tensors.id3x3_);  // value irrelevant at this point
  time_step_quantities_.last_substep_plastic_defgrad_inverse_.resize(
      1, const_non_mat_tensors.id3x3_);
  time_step_quantities_.last_elastic_stretch_eigenval_.resize(
      1, std::array<double, 3>{1.0, 1.0, 1.0});


  // update last_ and current_ values of the plastic strain
  time_step_quantities_.last_plastic_strain_.resize(1, 0.0);
  time_step_quantities_.current_plastic_strain_.resize(1, 0.0);  // value irrelevant at this point
  time_step_quantities_.last_substep_plastic_strain_.resize(1, 0.0);

  // update last_ plastic strain increment
  time_step_quantities_.last_plastic_strain_increment_.resize(1.0, 0.0);



  // update last_ and current_ values of the equivalent stress
  time_step_quantities_.last_equiv_stress_.resize(1.0, 0.0);
  time_step_quantities_.last_equiv_stress_elastic_pred_.resize(1.0, 0.0);
  time_step_quantities_.last_equiv_stress_plastic_pred_.resize(1.0, 0.0);
  time_step_quantities_.current_equiv_stress_.resize(1.0, 0.0);  // value irrelevant at this point

  // default values of the right CG tensor: unit tensor
  time_step_quantities_.last_rightCG_.resize(1, const_non_mat_tensors.id3x3_);
  time_step_quantities_.current_rightCG_.resize(
      1, const_non_mat_tensors.id3x3_);  // value irrelevant at this point

  // default value of the last deformation gradient: unit tensor
  time_step_quantities_.last_defgrad_.resize(1, const_non_mat_tensors.id3x3_);

  // default value for the current deformation gradient: zero tensor \f$ \boldsymbol{0} f$
  // (to make sure that the inverse inelastic deformation gradient is evaluated in the first
  // method call)
  time_step_quantities_.current_defgrad_.resize(
      1, Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});

  // general local time integration analysis: initialize csv writer
  if (parameter()->analyze_timint()) general_local_timint_analysis_utils.init_csv_writer();
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::pre_evaluate(
    const Teuchos::ParameterList& params, const EvaluationContext& context, const int gp,
    const int eleGID)
{
  // save parameter list
  params_ = params;

  // set Gauss Point
  gp_ = gp;

  // set element ID
  ele_gid_ = eleGID;

  // set time step
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  time_step_tracker_.dt_ = *context.time_step_size;
  FOUR_C_ASSERT(context.total_time, "Total time not given in evaluation context.");
  time_step_tracker_.tnp_ = *context.total_time;

  // set minimum substep length
  time_step_tracker_.min_dt_ =
      time_step_tracker_.dt_ / std::pow(2.0, parameter()->max_halve_number());

  // general local timint analysis utilities: save time step and time
  if (parameter()->analyze_timint())
  {
    FOUR_C_ASSERT_ALWAYS(ele_gid_ == 0,
        "We only want to use the time integration analysis for 1D simulations employing a "
        "single "
        "element! Your current element id is {}",
        ele_gid_);

    general_local_timint_analysis_utils.sim_timestep_ = time_step_tracker_.dt_;
    general_local_timint_analysis_utils.sim_time_ = time_step_tracker_.tnp_;
    // reset current time step if required
    if (!general_local_timint_analysis_utils.reset_called_)
    {
      general_local_timint_analysis_utils.reset();
      general_local_timint_analysis_utils.reset_called_ = true;
    }
  }

  // set last substep values (last converged state) as the last time step values -->
  // required, as these are used in the EvaluateAdditionalCMat method (in the case where
  // there is no plastic deformation, these would not be updated correctly otherwise)
  time_step_quantities_.last_substep_plastic_defgrad_inverse_[gp_] =
      time_step_quantities_.last_plastic_defgrad_inverse_[gp_];
  time_step_quantities_.last_substep_plastic_strain_[gp_] =
      time_step_quantities_.last_plastic_strain_[gp_];
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::prepare_return_mapping(
    const Core::LinAlg::Matrix<3, 3>& defgrad)
{
  // set current evaluation gp for the viscoplastic law
  viscoplastic_law_->pre_evaluate(params_, gp_, ele_gid_);  // set last_substep <- last_

  // set initial interpolation factors /  starting points for the Local Newton Guess
  // Interpolation routine
  if (parameter()->use_lngi())
  {
    prepare_lngi(defgrad);
  }
  // set LNL iteration to 0
  lnl_data_.iter_ = 0;

  // Increment the global iteration here (only for first GP, we don't want to do this for
  // each GP). We assume that this method is only called in new global iterations!
  if (gp_ == 0) ++globiter_;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::calculate_gamma_delta(
    const Core::LinAlg::Matrix<3, 3>& CeM, Core::LinAlg::Matrix<3, 1>& gamma,
    Core::LinAlg::Matrix<8, 1>& delta)
{
  // compute principal values
  Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> CeV_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(CeM, CeV_strain);
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv, CeV_strain);

  // compute derivatives of principle invariants
  Core::LinAlg::Matrix<3, 1> dPIe(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> ddPIIe(Core::LinAlg::Initialization::zero);
  // clear variables
  dPIe.clear();
  ddPIIe.clear();

  // loop over map of associated potential summands
  // derivatives of strain energy function w.r.t. principal invariants
  for (auto& p : potsumel_)  // only isotropic
  {
    p->add_derivatives_principal(dPIe, ddPIIe, prinv, gp_, ele_gid_);
  }

  // compose coefficients
  Mat::calculate_gamma_delta(gamma, delta, prinv, dPIe, ddPIIe);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
StateQuantities Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_state_quantities(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<3, 3>& iFinM,
    const double plastic_strain, ErrorType& err_status, const double dt,
    const StateQuantityEvalType& eval_type)
{
  StateQuantities state_quantities{};

  // auxiliaries
  Core::LinAlg::Matrix<1, 1> temp1x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 3> temp1x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> temp3x3;

  // compute right elastic CG tensor
  temp3x3.multiply_nn(1.0, CM, iFinM, 0.0);
  state_quantities.curr_CeM_.multiply_tn(1.0, iFinM, temp3x3, 0.0);
  Core::LinAlg::SymmetricTensor<double, 3, 3> CeV =
      Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(state_quantities.curr_CeM_));

  // compose isotropic elastic coefficients (Holzapfel, Nonlinear Solid Mechanics, 2000)
  calculate_gamma_delta(
      state_quantities.curr_CeM_, state_quantities.curr_gamma_, state_quantities.curr_delta_);
  state_quantities.curr_SeM_.clear();
  state_quantities.curr_dSedCe_.clear();
  // compute additional 2nd elastic PK stress and elastic stiffness for the transversely
  // isotropic components (additive split assumed, as for CoupTransverselyIsotropic)
  if (parameter()->mat_behavior() == MatBehavior::transv_isotrop)
  {
    // initialize empty parameter list
    Teuchos::ParameterList param_list{};

    // loop through all transversely isotropic parts, and compute the additional elastic
    // stress and elastic stiffness
    Core::LinAlg::SymmetricTensor<double, 3, 3> SeV{};

    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> dSedCe{};
    for (auto& p : potsumel_transviso_)
    {
      p->add_stress_aniso_principal(CeV, dSedCe, SeV, param_list, gp_, ele_gid_);
    }
    state_quantities.curr_dSedCe_.clear();
    state_quantities.curr_dSedCe_ += Core::LinAlg::make_stress_like_voigt_view(dSedCe);
    state_quantities.curr_SeM_ = Core::LinAlg::make_matrix(Core::LinAlg::get_full(SeV));
  }

  // Ce * Ce tensor
  Core::LinAlg::Matrix<3, 3> CeCeM(Core::LinAlg::Initialization::zero);
  CeCeM.multiply_nn(1.0, state_quantities.curr_CeM_, state_quantities.curr_CeM_, 0.0);

  // compute symmetric part of Mandel stress tensor
  Core::LinAlg::Matrix<3, 3> Me_sym_M(Core::LinAlg::Initialization::zero);
  Me_sym_M.update(state_quantities.curr_gamma_(0), state_quantities.curr_CeM_,
      state_quantities.curr_gamma_(1), CeCeM, 0.0);
  Me_sym_M.update(state_quantities.curr_gamma_(2), const_non_mat_tensors.id3x3_, 1.0);
  if (parameter()->mat_behavior() == MatBehavior::transv_isotrop)
  {
    Core::LinAlg::Matrix<3, 3> addMeM(Core::LinAlg::Initialization::zero);
    temp3x3.multiply_nn(1.0, state_quantities.curr_CeM_, state_quantities.curr_SeM_, 0.0);
    addMeM.update(1.0 / 2.0, temp3x3, 0.0);
    temp3x3.multiply_tn(1.0, state_quantities.curr_SeM_, state_quantities.curr_CeM_, 0.0);
    addMeM.update(1.0 / 2.0, temp3x3, 1.0);
    Me_sym_M.update(1.0, addMeM, 1.0);
  }

  // calculate deviatoric part of the symmetric Mandel stress
  double trMe = 0.0;
  for (int i = 0; i < 3; ++i) trMe += Me_sym_M(i, i);
  state_quantities.curr_Me_dev_sym_M_.update(
      1.0, Me_sym_M, -1.0 / 3.0 * trMe, const_non_mat_tensors.id3x3_, 0.0);

  // for transverse isotropy: we use the Hill 1949 yield condition, adapted for transversely
  // isotropic materials --> get yield function parameters A, B, F
  const double A = parameter()->yield_cond_a();
  const double B = parameter()->yield_cond_b();
  const double F = parameter()->yield_cond_f();

  // determine scalar quantities of invariants / pseudoinvariants needed to compute the
  // equivalent tensile stress
  double Me_dev_sym_contract_Me_dev_sym =
      Core::LinAlg::FourTensorOperations::contract_matrix_matrix(
          state_quantities.curr_Me_dev_sym_M_, state_quantities.curr_Me_dev_sym_M_);
  Core::LinAlg::Matrix<3, 3> Me_dev_sym_squared_M(Core::LinAlg::Initialization::zero);
  Me_dev_sym_squared_M.multiply_nn(
      1.0, state_quantities.curr_Me_dev_sym_M_, state_quantities.curr_Me_dev_sym_M_, 0.0);
  temp1x3.multiply_tn(1.0, m_, Me_dev_sym_squared_M, 0.0);
  temp1x1.multiply_nn(1.0, temp1x3, m_, 0.0);
  double mTMe_dev_sym_squared_m = temp1x1(0);
  temp1x3.multiply_tn(1.0, m_, state_quantities.curr_Me_dev_sym_M_, 0.0);
  temp1x1.multiply_nn(1.0, temp1x3, m_, 0.0);
  double mTMe_dev_sym_m = temp1x1(0);

  // calculate equivalent tensile stress
  if (parameter()->mat_behavior() == MatBehavior::transv_isotrop)
  {
    state_quantities.curr_equiv_stress_ =
        std::sqrt((A + 2 * B) * Me_dev_sym_contract_Me_dev_sym +
                  2 * (F - A - 2 * B) * mTMe_dev_sym_squared_m +
                  (5 * A + B - 2 * F) * std::pow(mTMe_dev_sym_m, 2.0));
  }
  else
  {
    state_quantities.curr_equiv_stress_ = std::sqrt(3.0 / 2.0 * Me_dev_sym_contract_Me_dev_sym);
  }

  // check if either stress or plastic strain are NaN -> in this case,
  // return with overflow error
  if (std::isnan(state_quantities.curr_equiv_stress_) || std::isnan(plastic_strain))
  {
    // return with error
    err_status = ErrorType::overflow_error;
    return state_quantities;
  }

  if (eval_type ==
      InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityEvalType::EquivStressOnly)
  {
    return state_quantities;
  }


  // calculate equivalent plastic strain rate using the viscoplastic law
  state_quantities.curr_equiv_plastic_strain_rate_ =
      viscoplastic_law_->evaluate_plastic_strain_rate(state_quantities.curr_equiv_stress_,
          plastic_strain, dt, parameter()->max_plastic_strain_incr(), err_status, update_hist_var_);

  if (eval_type == StateQuantityEvalType::PlasticStrainRateOnly)
  {
    return state_quantities;
  }

  // return if we get an error, all other calculations are useless since substepping is
  // triggered
  if (err_status != ErrorType::no_errors)
  {
    // return with error
    return state_quantities;
  }

  // calculate plastic flow direction
  if (parameter()->mat_behavior() == MatBehavior::transv_isotrop)
  {
    // determine required components for the computation of the plastic flow direction
    Core::LinAlg::Matrix<3, 1> Me_dev_sym_m(Core::LinAlg::Initialization::zero);
    Me_dev_sym_m.multiply_nn(1.0, state_quantities.curr_Me_dev_sym_M_, m_, 0.0);
    Core::LinAlg::Matrix<3, 3> m_dyad_Me_dev_sym_m(Core::LinAlg::Initialization::zero);
    m_dyad_Me_dev_sym_m.multiply_nt(1.0, m_, Me_dev_sym_m, 0.0);
    Core::LinAlg::Matrix<3, 3> Me_dev_sym_A_dyad_A(Core::LinAlg::Initialization::zero);
    Me_dev_sym_A_dyad_A.multiply_nt(1.0, Me_dev_sym_m, m_, 0.0);

    state_quantities.curr_NpM_.clear();
    state_quantities.curr_dpM_.clear();
    if (state_quantities.curr_equiv_stress_ > 0.0)
    {
      // build the plastic flow direction from its tensor parts
      state_quantities.curr_NpM_.update(
          -2.0 / (3.0 * state_quantities.curr_equiv_stress_) * (F - A - 2.0 * B) * mTMe_dev_sym_m,
          const_non_mat_tensors.id3x3_, 0.0);
      state_quantities.curr_NpM_.update(
          1.0 / (1.0 * state_quantities.curr_equiv_stress_) * (A + 2.0 * B),
          state_quantities.curr_Me_dev_sym_M_, 1.0);
      state_quantities.curr_NpM_.update(
          1.0 / (2.0 * state_quantities.curr_equiv_stress_) * 2.0 * (F - A - 2.0 * B),
          m_dyad_Me_dev_sym_m, 1.0);
      state_quantities.curr_NpM_.update(
          1.0 / (2.0 * state_quantities.curr_equiv_stress_) * 2.0 * (F - A - 2.0 * B),
          Me_dev_sym_A_dyad_A, 1.0);
      state_quantities.curr_NpM_.update(1.0 / (2.0 * state_quantities.curr_equiv_stress_) *
                                            (5.0 * A + B - 2.0 * F) * 2.0 * mTMe_dev_sym_m,
          const_mat_tensors_.mm_dev_, 1.0);

      // calculate plastic stretching tensor (deformation rate tensor)
      state_quantities.curr_dpM_.update(
          state_quantities.curr_equiv_plastic_strain_rate_, state_quantities.curr_NpM_, 0.0);
    }

    // calculate plastic velocity gradient tensor
    state_quantities.curr_lpM_.multiply_nn(
        1.0, const_mat_tensors_.id_plus_mm_, state_quantities.curr_dpM_, 0.0);
    state_quantities.curr_lpM_.multiply_nn(
        -1.0, state_quantities.curr_dpM_, const_mat_tensors_.mm_, 1.0);
  }
  else
  {
    state_quantities.curr_NpM_.clear();
    state_quantities.curr_dpM_.clear();
    if (state_quantities.curr_equiv_stress_ > 0.0)
    {
      // build the plastic flow direction from its tensor parts
      state_quantities.curr_NpM_.update(3.0 / (2.0 * state_quantities.curr_equiv_stress_),
          state_quantities.curr_Me_dev_sym_M_, 0.0);

      // calculate plastic stretching tensor (deformation rate tensor)
      state_quantities.curr_dpM_.update(
          state_quantities.curr_equiv_plastic_strain_rate_, state_quantities.curr_NpM_, 0.0);
    }

    // calculate plastic velocity gradient tensor
    state_quantities.curr_lpM_.update(1.0, state_quantities.curr_dpM_, 0.0);
  }

  // calculate plastic update tensor (only required, and computed, for
  // standard time integration)
  if (parameter()->timint_type() == TimIntType::standard)
  {
    temp3x3.update(-dt, state_quantities.curr_lpM_, 0.0);
    Core::LinAlg::MatrixFunctErrorType exp_err_status =
        Core::LinAlg::MatrixFunctErrorType::no_errors;
    state_quantities.curr_EpM_ =
        Core::LinAlg::matrix_exp(temp3x3, exp_err_status, parameter()->mat_exp_calc_method());
    if (exp_err_status != Core::LinAlg::MatrixFunctErrorType::no_errors)
    {
      err_status = ErrorType::failed_matrix_exp_evaluation;
      return state_quantities;
    }
  }

  return state_quantities;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
StateQuantityDerivatives
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_state_quantity_derivatives(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<3, 3>& iFinM,
    const double plastic_strain, ErrorType& err_status, const double dt,
    const StateQuantityDerivEvalType& eval_type, const bool eval_state)
{
  StateQuantityDerivatives state_quantity_derivatives{};

  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 9> temp6x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 6> temp9x6(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);

  // check whether to reevaluate the state or to keep the available state quantity values
  StateQuantities relevant_state_quantities = state_quantities_;
  if (eval_state)
  {
    relevant_state_quantities = evaluate_state_quantities(
        CM, iFinM, plastic_strain, err_status, dt, StateQuantityEvalType::FullEval);
  }

  // get the state quantities
  const Core::LinAlg::Matrix<3, 3> CeM = relevant_state_quantities.curr_CeM_;
  const Core::LinAlg::Matrix<3, 1> gamma = relevant_state_quantities.curr_gamma_;
  const Core::LinAlg::Matrix<8, 1> delta = relevant_state_quantities.curr_delta_;
  const Core::LinAlg::Matrix<3, 3> SeM = relevant_state_quantities.curr_SeM_;
  const Core::LinAlg::Matrix<6, 6> dSedCe = relevant_state_quantities.curr_dSedCe_;
  const Core::LinAlg::Matrix<3, 3> Me_dev_sym_M = relevant_state_quantities.curr_Me_dev_sym_M_;
  const double equiv_stress = relevant_state_quantities.curr_equiv_stress_;
  const double equiv_plastic_strain_rate =
      relevant_state_quantities.curr_equiv_plastic_strain_rate_;
  const Core::LinAlg::Matrix<3, 3> NpM = relevant_state_quantities.curr_NpM_;
  const Core::LinAlg::Matrix<3, 3> dpM = relevant_state_quantities.curr_dpM_;
  const Core::LinAlg::Matrix<3, 3> lpM = relevant_state_quantities.curr_lpM_;
  const Core::LinAlg::Matrix<3, 3> EpM = relevant_state_quantities.curr_EpM_;


  // compute the relevant derivatives of the elastic right Cauchy-Green deformation tensor
  Mat::elast_hyper_get_derivs_of_elastic_right_cg_tensor(Core::LinAlg::make_tensor(iFinM),
      Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(CM)),
      state_quantity_derivatives.curr_dCedC_, state_quantity_derivatives.curr_dCediFin_);
  // save these also as four tensors
  Core::LinAlg::FourTensor<3> dCediFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(
      dCediFin_FourTensor, state_quantity_derivatives.curr_dCediFin_);
  Core::LinAlg::FourTensor<3> dCedC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(
      dCedC_FourTensor, state_quantity_derivatives.curr_dCedC_);

  // inverse inelastic right Cauchy-Green deformation tensor
  Core::LinAlg::Matrix<3, 3> iCinM(Core::LinAlg::Initialization::zero);
  iCinM.multiply_nt(1.0, iFinM, iFinM, 0.0);
  Core::LinAlg::Matrix<6, 1> iCinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinM, iCinV);

  // elastic right Cauchy-Green tensor in stress form
  Core::LinAlg::Matrix<6, 1> CeV(Core::LinAlg::Initialization::zero);  // stress-form
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(CeM, CeV);

  // inverse elastic right Cauchy-Green tensor
  Core::LinAlg::Matrix<3, 3> iCeM(Core::LinAlg::Initialization::zero);
  iCeM.invert(CeM);
  Core::LinAlg::Matrix<6, 1> iCeV(Core::LinAlg::Initialization::zero);  // stress-form
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCeM, iCeV);

  // inverse transposed inelastic defgrad
  Core::LinAlg::Matrix<3, 3> iFinTM(Core::LinAlg::Initialization::zero);
  iFinTM.multiply_tn(1.0, iFinM, const_non_mat_tensors.id3x3_, 0.0);

  // calculate various other helper tensors required for subsequent computation
  Core::LinAlg::Matrix<3, 3> CiFinM(Core::LinAlg::Initialization::zero);
  CiFinM.multiply_nn(1.0, CM, iFinM, 0.0);
  Core::LinAlg::Matrix<9, 1> CiFinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFinM, CiFinV);

  Core::LinAlg::Matrix<3, 3> iFinTCM(Core::LinAlg::Initialization::zero);
  iFinTCM.multiply_tn(1.0, iFinM, CM, 0.0);

  Core::LinAlg::Matrix<3, 3> CeiFinTCM(Core::LinAlg::Initialization::zero);
  temp3x3.multiply_nt(1.0, CeM, iFinM, 0.0);
  CeiFinTCM.multiply_nn(1.0, temp3x3, CM, 0.0);

  Core::LinAlg::Matrix<3, 3> CiFinCeM(Core::LinAlg::Initialization::zero);
  CiFinCeM.multiply_nn(1.0, CiFinM, CeM, 0.0);
  Core::LinAlg::Matrix<9, 1> CiFinCeV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFinCeM, CiFinCeV);

  Core::LinAlg::Matrix<3, 3> CeCeM(Core::LinAlg::Initialization::zero);
  CeCeM.multiply_nn(1.0, CeM, CeM, 0.0);
  Core::LinAlg::Matrix<6, 1> CeCeV(Core::LinAlg::Initialization::zero);  // stress-form
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(CeCeM, CeCeV);

  Core::LinAlg::Matrix<3, 3> CiFiniCeM(Core::LinAlg::Initialization::zero);
  CiFiniCeM.multiply_nn(1.0, CiFinM, iCeM, 0.0);
  Core::LinAlg::Matrix<9, 1> CiFiniCeV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFiniCeM, CiFiniCeV);

  Core::LinAlg::Matrix<3, 3> CeiFinTM(Core::LinAlg::Initialization::zero);
  CeiFinTM.multiply_nn(1.0, CeM, iFinTM, 0.0);

  Core::LinAlg::Matrix<3, 3> iCinCiCinM(Core::LinAlg::Initialization::zero);
  temp3x3.multiply_nn(1.0, CM, iCinM, 0.0);
  iCinCiCinM.multiply_nn(1.0, iCinM, temp3x3, 0.0);
  Core::LinAlg::Matrix<6, 1> iCinCiCinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinCiCinM, iCinCiCinV);

  Core::LinAlg::Matrix<3, 3> iCM(Core::LinAlg::Initialization::zero);
  iCM.invert(CM);
  Core::LinAlg::Matrix<6, 1> iCV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCM, iCV);

  // compute the relevant derivatives of the symmetric part of the Mandel stress

  // \f$ \frac{\partial \boldsymbol{M}^{\text{e}}_{\text{sym}} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  Core::LinAlg::Matrix<6, 9> dMe_sym_diFin(Core::LinAlg::Initialization::zero);
  dMe_sym_diFin.clear();
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMe_sym_diFin, iFinTCM, const_non_mat_tensors.id3x3_, gamma(0));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMe_sym_diFin, iFinTCM, CeM, gamma(1));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMe_sym_diFin, CeiFinTCM, const_non_mat_tensors.id3x3_, gamma(1));
  dMe_sym_diFin.multiply_nt(delta(0), CeV, CiFinV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(1), CeV, CiFinCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(1), CeCeV, CiFinV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(2), CeV, CiFiniCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(2), const_non_mat_tensors.id6x1_, CiFinV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(3), CeCeV, CiFinCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(4), CeCeV, CiFiniCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(4), const_non_mat_tensors.id6x1_, CiFinCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(5), const_non_mat_tensors.id6x1_, CiFiniCeV, 1.0);

  // \f$ \frac{\partial \boldsymbol{M}^{\text{e}}_{\text{sym}} }{\partial
  // \boldsymbol{C}^{}_{}} \f$ (Voigt stress-stress form)
  Core::LinAlg::Matrix<6, 6> dMe_sym_dC(Core::LinAlg::Initialization::zero);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMe_sym_dC, gamma(0), iFinTM, iFinTM, 0.0);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMe_sym_dC, gamma(1), iFinTM, CeiFinTM, 1.0);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMe_sym_dC, gamma(1), CeiFinTM, iFinTM, 1.0);
  dMe_sym_dC.multiply_nt(delta(0) / 2.0, CeV, iCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(1) / 2.0, CeV, iCinCiCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(1) / 2.0, CeCeV, iCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(2) / 2.0, CeV, iCV, 1.0);
  dMe_sym_dC.multiply_nt(delta(2) / 2.0, const_non_mat_tensors.id6x1_, iCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(3) / 2.0, CeCeV, iCinCiCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(4) / 2.0, CeCeV, iCV, 1.0);
  dMe_sym_dC.multiply_nt(delta(4) / 2.0, const_non_mat_tensors.id6x1_, iCinCiCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(5) / 2.0, const_non_mat_tensors.id6x1_, iCV, 1.0);

  // compute derivative of the additional transversely isotropic stress (w.r.t. right
  // elastic Cauchy-Green deformation tensor) in stress-strain notation
  temp6x6.update(1.0, dSedCe, 0.0);
  Core::LinAlg::Matrix<6, 6> dSedCe_stress_strain =
      Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // \f$ \frac{\partial \boldsymbol{S}^{\text{e}}_{\text{trn}} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  Core::LinAlg::Matrix<6, 9> dSediFin(Core::LinAlg::Initialization::zero);
  dSediFin.multiply_nn(1.0, dSedCe_stress_strain, state_quantity_derivatives.curr_dCediFin_, 0.0);
  Core::LinAlg::FourTensor<3> dSediFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(dSediFin_FourTensor, dSediFin);

  // \f$ \frac{\partial \boldsymbol{S}^{\text{e}}_{\text{trn}} }{\partial
  // \boldsymbol{C}^{}_{}} \f$ (Voigt stress-stress form)
  Core::LinAlg::Matrix<6, 6> dSedC(Core::LinAlg::Initialization::zero);
  dSedC.multiply_nn(1.0, dSedCe_stress_strain, state_quantity_derivatives.curr_dCedC_, 0.0);
  Core::LinAlg::FourTensor<3> dSedC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(dSedC_FourTensor, dSedC);

  // compute additional components of the elastic transversely isotropic components for the
  // derivatives of the symmetric Mandel stress
  if (parameter()->mat_behavior() == MatBehavior::transv_isotrop)
  {
    Core::LinAlg::FourTensor<3> CedSediFin_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        CedSediFin_FourTensor, CeM, dSediFin_FourTensor, true);
    Core::LinAlg::FourTensor<3> CedSediFin_T12_FourTensor(true);
    CedSediFin_T12_FourTensor.transpose_12(CedSediFin_FourTensor);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(temp6x9, CedSediFin_FourTensor);
    dMe_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(
        temp6x9, CedSediFin_T12_FourTensor);
    dMe_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::FourTensor<3> SedCediFin_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        SedCediFin_FourTensor, SeM, dCediFin_FourTensor, true);
    Core::LinAlg::FourTensor<3> SedCediFin_T12_FourTensor(true);
    SedCediFin_T12_FourTensor.transpose_12(SedCediFin_FourTensor);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(temp6x9, SedCediFin_FourTensor);
    dMe_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(
        temp6x9, SedCediFin_T12_FourTensor);
    dMe_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);

    Core::LinAlg::FourTensor<3> CedSedC_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        CedSedC_FourTensor, CeM, dSedC_FourTensor, true);
    Core::LinAlg::FourTensor<3> CedSedC_T12_FourTensor(true);
    CedSedC_T12_FourTensor.transpose_12(CedSedC_FourTensor);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, CedSedC_FourTensor);
    dMe_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, CedSedC_T12_FourTensor);
    dMe_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::FourTensor<3> SedCedC_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        SedCedC_FourTensor, SeM, dCedC_FourTensor, true);
    Core::LinAlg::FourTensor<3> SedCedC_T12_FourTensor(true);
    SedCedC_T12_FourTensor.transpose_12(SedCedC_FourTensor);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, SedCedC_FourTensor);
    dMe_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, SedCedC_T12_FourTensor);
    dMe_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
  }

  // compute derivatives of the deviatoric, symmetric part of the Mandel stress

  // \f$ \frac{\partial \boldsymbol{M}^{\text{e}}_{\text{dev, sym}} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  state_quantity_derivatives.curr_dMe_dev_sym_diFin_.multiply_nn(
      1.0, const_non_mat_tensors.dev_op_, dMe_sym_diFin, 0.0);
  // \f$ \frac{\partial \boldsymbol{M}^{\text{e}}_{\text{dev,sym}} }{\partial
  // \boldsymbol{C}^{}_{}} \f$ (Voigt stress-stress form)
  state_quantity_derivatives.curr_dMe_dev_sym_dC_.multiply_nn(
      1.0, const_non_mat_tensors.dev_op_, dMe_sym_dC, 0.0);

  // plastic flow direction in Voigt strain notation
  Core::LinAlg::Matrix<6, 1> NpV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(NpM, NpV);

  // compute derivatives of the equivalent stress

  // \f$ \frac{\partial \overline{\sigma} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  state_quantity_derivatives.curr_dequiv_stress_diFin_.multiply_tn(
      1.0, NpV, state_quantity_derivatives.curr_dMe_dev_sym_diFin_, 0.0);
  // \f$ \frac{\partial \overline{\sigma} }{\partial
  // \boldsymbol{C}^{}} \f$ (Voigt stress-form)
  state_quantity_derivatives.curr_dequiv_stress_dC_.multiply_tn(
      1.0, NpV, state_quantity_derivatives.curr_dMe_dev_sym_dC_, 0.0);

  // recompute flow direction in stress form
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(NpM, NpV);

  // we use the Hill 1949 yield condition, adapted for transversely isotropic materials ->
  // get yield condition parameters A, B, and F
  const double A = parameter()->yield_cond_a();
  const double B = parameter()->yield_cond_b();
  const double F = parameter()->yield_cond_f();

  // compute required derivative of the plastic flow direction (w.r.t. dev., sym. part of
  // the Mandel stress)
  // \f$ \frac{\partial \boldsymbol{N}^{\text{p}}_{} }{\partial
  // \partial \boldsymbol{M}^{\text{e}}_{\text{dev,sym}}} \f$ (Voigt stress-stress form)
  Core::LinAlg::Matrix<6, 6> dNpdMe_sym_dev(Core::LinAlg::Initialization::zero);
  if (parameter()->mat_behavior() == MatBehavior::transv_isotrop)
  {
    dNpdMe_sym_dev.multiply_nt(-1.0 / equiv_stress, NpV, NpV, 0.0);
    dNpdMe_sym_dev.update(-1.0 / 2.0 * 1.0 / equiv_stress * 4.0 / 3.0 * (F - A - 2.0 * B),
        const_mat_tensors_.id_dyad_mm_, 1.0);
    dNpdMe_sym_dev.update(
        1.0 / 1.0 * 1.0 / equiv_stress * (A + 2 * B), const_non_mat_tensors.id4_6x6_, 1.0);
    Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(dNpdMe_sym_dev,
        1.0 / equiv_stress * (F - A - 2 * B), const_mat_tensors_.mm_, const_non_mat_tensors.id3x3_,
        1.0);
    Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(dNpdMe_sym_dev,
        1.0 / equiv_stress * (F - A - 2 * B), const_non_mat_tensors.id3x3_, const_mat_tensors_.mm_,
        1.0);
    dNpdMe_sym_dev.update(
        1.0 / equiv_stress * (5 * A + B - 2 * F), const_mat_tensors_.mm_dev_dyad_mm_, 1.0);
  }
  else
  {
    dNpdMe_sym_dev.multiply_nt(-1.0 / equiv_stress, NpV, NpV, 0.0);
    dNpdMe_sym_dev.update(1.0 / equiv_stress * 3.0 / 2.0, const_non_mat_tensors.id4_6x6_, 1.0);
  }
  // convert derivative to Voigt stress-strain form
  temp6x6 = dNpdMe_sym_dev;
  dNpdMe_sym_dev = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // compute the relevant derivatives of the plastic strain rate
  Core::LinAlg::Matrix<2, 1> evoEqFunctionDers =
      viscoplastic_law_->evaluate_derivatives_of_plastic_strain_rate(equiv_stress, plastic_strain,
          dt, parameter()->max_plastic_strain_deriv_incr(), err_status);

  // return if we get an error, all other calculations are useless since substepping is
  // triggered
  if (err_status != ErrorType::no_errors)
  {
    // return with error
    return StateQuantityDerivatives{};
  }

  if (eval_type == InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivEvalType::
                       EquivStressDerivsOnly)
  {
    return state_quantity_derivatives;
  }

  // compute derivatives of the plastic strain rate
  state_quantity_derivatives.curr_dpsr_dequiv_stress_ = evoEqFunctionDers(0);
  state_quantity_derivatives.curr_dpsr_depsp_ = evoEqFunctionDers(1);


  if (eval_type == InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivEvalType::
                       PlasticStrainRateDerivsOnly)
  {
    return state_quantity_derivatives;
  }


  // compute derivatives of the plastic stretching tensor...
  Core::LinAlg::Matrix<6, 6> Np_dyad_Np_V(
      Core::LinAlg::Initialization::zero);  // in stress-strain form
  temp6x6.multiply_nt(1.0, NpV, NpV, 0.0);
  Np_dyad_Np_V = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);
  temp6x6.update(state_quantity_derivatives.curr_dpsr_dequiv_stress_, Np_dyad_Np_V,
      equiv_plastic_strain_rate, dNpdMe_sym_dev, 0.0);

  // ... w.r.t. invese inelastic defgrad
  state_quantity_derivatives.curr_ddpdiFin_.multiply_nn(
      1.0, temp6x6, state_quantity_derivatives.curr_dMe_dev_sym_diFin_, 0.0);
  // ... w.r.t. plastic strain
  state_quantity_derivatives.curr_ddpdepsp_.update(
      state_quantity_derivatives.curr_dpsr_depsp_, NpV, 0.0);
  // ... w.r.t. right CG
  state_quantity_derivatives.curr_ddpdC_.multiply_nn(
      1.0, temp6x6, state_quantity_derivatives.curr_dMe_dev_sym_dC_, 0.0);

  // compute derivatives of the plastic velocity gradient ...

  // ... w.r.t. invese inelastic defgrad
  Core::LinAlg::FourTensor<3> ddpdiFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(
      ddpdiFin_FourTensor, state_quantity_derivatives.curr_ddpdiFin_);
  Core::LinAlg::FourTensor<3> id_plus_mm_ddpdiFin_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      id_plus_mm_ddpdiFin_FourTensor, const_mat_tensors_.id_plus_mm_, ddpdiFin_FourTensor, true);
  Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(
      temp9x9, id_plus_mm_ddpdiFin_FourTensor);
  state_quantity_derivatives.curr_dlpdiFin_.update(1.0, temp9x9, 0.0);
  Core::LinAlg::FourTensor<3> mm_ddpdiFin_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      mm_ddpdiFin_FourTensor, const_mat_tensors_.mm_, ddpdiFin_FourTensor, true);
  Core::LinAlg::FourTensor<3> mm_ddpdiFin_T12_FourTensor(true);
  mm_ddpdiFin_T12_FourTensor.transpose_12(mm_ddpdiFin_FourTensor);
  Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, mm_ddpdiFin_T12_FourTensor);
  state_quantity_derivatives.curr_dlpdiFin_.update(-1.0, temp9x9, 1.0);

  // ... w.r.t. right CG
  Core::LinAlg::FourTensor<3> ddpdC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(
      ddpdC_FourTensor, state_quantity_derivatives.curr_ddpdC_);
  Core::LinAlg::FourTensor<3> id_plus_mm_ddpdC_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      id_plus_mm_ddpdC_FourTensor, const_mat_tensors_.id_plus_mm_, ddpdC_FourTensor, true);
  Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(
      temp9x6, id_plus_mm_ddpdC_FourTensor);
  state_quantity_derivatives.curr_dlpdC_.update(1.0, temp9x6, 0.0);
  Core::LinAlg::FourTensor<3> mm_ddpdC_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      mm_ddpdC_FourTensor, const_mat_tensors_.mm_, ddpdC_FourTensor, true);
  Core::LinAlg::FourTensor<3> mm_ddpdC_T12_FourTensor(true);
  mm_ddpdC_T12_FourTensor.transpose_12(mm_ddpdC_FourTensor);
  Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(temp9x6, mm_ddpdC_T12_FourTensor);
  state_quantity_derivatives.curr_dlpdC_.update(-1.0, temp9x6, 1.0);

  // ... w.r.t. plastic strain
  Core::LinAlg::Matrix<3, 3> ddpdepsp_M(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(
      state_quantity_derivatives.curr_ddpdepsp_, ddpdepsp_M);
  Core::LinAlg::Matrix<3, 3> dlpdepsp_M(Core::LinAlg::Initialization::zero);
  dlpdepsp_M.multiply_nn(1.0, const_mat_tensors_.id_plus_mm_, ddpdepsp_M, 0.0);
  dlpdepsp_M.multiply_nn(-1.0, ddpdepsp_M, const_mat_tensors_.mm_, 1.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(dlpdepsp_M, state_quantity_derivatives.curr_dlpdepsp_);


  // compute derivatives of the update tensor (only required for standard substepping)
  if (parameter()->timint_type() == TimIntType::standard)
  {
    // compute argument
    Core::LinAlg::Matrix<3, 3> min_dt_lpM(Core::LinAlg::Initialization::zero);
    min_dt_lpM.update(-1.0 * dt, lpM, 0.0);

    // compute derivative of exponential ...

    // ... w.r.t. its argument
    Core::LinAlg::MatrixFunctErrorType exp_err_status =
        Core::LinAlg::MatrixFunctErrorType::no_errors;
    Core::LinAlg::Matrix<9, 9> expderivV = Core::LinAlg::matrix_3x3_exp_1st_deriv(
        min_dt_lpM, exp_err_status, parameter()->mat_exp_deriv_calc_method());
    if (exp_err_status != Core::LinAlg::MatrixFunctErrorType::no_errors)
    {
      err_status = ErrorType::failed_matrix_exp_evaluation;
      return state_quantity_derivatives;
    }

    // ... w.r.t. inverse inelastic defgrad
    state_quantity_derivatives.curr_dEpdiFin_.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdiFin_, 0.0);

    // ... w.r.t. right CG
    state_quantity_derivatives.curr_dEpdC_.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdC_, 0.0);

    // ... w.r.t. plastic strain
    state_quantity_derivatives.curr_dEpdepsp_.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdepsp_, 0.0);
  }

  return state_quantity_derivatives;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  // reduced deformation gradient FredM, taking into account all the already computed
  // inelastic factors
  //    \f$ \boldsymbol{F_{\text{red}}} = \boldsymbol{F}
  //    \boldsymbol{F_{\text{in,other}}^{-1}}
  //    \f$
  //      with \f$\boldsymbol{F}_{\text{in,other}}^{-1} = \boldsymbol{F}_{\text{in},1}^{-1}
  //      \boldsymbol{F}_{\text{in},2}^{-1} \dots \f$ up to the current inelastic factor
  Core::LinAlg::Matrix<3, 3> FredM(Core::LinAlg::Initialization::zero);
  FredM.multiply_nn(1.0, *defgrad, iFin_other, 0.0);


  // reduced right Cauchy-Green deformation tensor
  Core::LinAlg::Matrix<3, 3> CredM(Core::LinAlg::Initialization::zero);
  CredM.multiply_tn(1.0, FredM, FredM, 0.0);

  // auxiliaries
  Core::LinAlg::FourTensor<3> tempFourTensor(true);

  // declare error status (no errors)
  ErrorType err_status = ErrorType::no_errors;

  // recompute the state to make sure that everything is evaluated properly after
  // circumventing the stiffness evaluation
  state_quantities_ =
      evaluate_state_quantities(CredM, time_step_quantities_.current_plastic_defgrad_inverse_[gp_],
          time_step_quantities_.current_plastic_strain_[gp_], err_status, time_step_tracker_.dt_,
          StateQuantityEvalType::FullEval);

  // calculate linearization term only if we have plastic strain
  if (std::abs(state_quantities_.curr_equiv_plastic_strain_rate_) > 0.0)
  {
    // ----- perturbation-based linearization ----- //
    if (parameter()->linearization_type() == LinearizationType::perturbation_based)
    {
      evaluate_additional_cmat_perturb_based(FredM, cmatadd, iFin_other, dSdiFinj);

      return;
    }

    // ----- analytical linearization ----- //

    if (err_status != ErrorType::no_errors)
    {
      evaluate_additional_cmat_perturb_based(FredM, cmatadd, iFin_other, dSdiFinj);

      return;
    }



    // calculate Jacobian
    Core::LinAlg::Matrix<10, 1> current_sol =
        wrap_unknowns(time_step_quantities_.current_plastic_defgrad_inverse_[gp_],
            time_step_quantities_.current_plastic_strain_[gp_]);

    Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
    viscoplastic_law_->pre_evaluate(params_, gp_, ele_gid_);  // set last_substep <- last_
    jacMat = calculate_jacobian(CredM, current_sol,
        time_step_quantities_.last_plastic_defgrad_inverse_[gp_],
        time_step_quantities_.last_plastic_strain_[gp_], time_step_tracker_.dt_, err_status);

    if (err_status != ErrorType::no_errors)
    {
      evaluate_additional_cmat_perturb_based(FredM, cmatadd, iFin_other, dSdiFinj);

      return;
    }

    // if we get singular Jacobian: throw exception -> go to FD-based linearization
    if (abs(jacMat.determinant()) < 1.0e-10)
    {
      err_status = ErrorType::singular_jacobian;
      evaluate_additional_cmat_perturb_based(FredM, cmatadd, iFin_other, dSdiFinj);

      return;
    }

    // declare right-hand side (RHS) terms of the linear system of equations related to the
    // analytical linearization
    Core::LinAlg::Matrix<9, 6> rhs_iFin_V(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 6> rhs_epsp_V(Core::LinAlg::Initialization::zero);

    if (parameter()->timint_type() == TimIntType::standard)
    // standard time integration
    {
      // calculate RHS of the equation for the plastic deformation gradient
      Core::LinAlg::FourTensor<3> dEpdC_FourTensor(true);
      Core::LinAlg::Voigt::setup_four_tensor_from_9x6_voigt_matrix(
          dEpdC_FourTensor, state_quantity_derivatives_.curr_dEpdC_);
      Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(tempFourTensor,
          time_step_quantities_.last_plastic_defgrad_inverse_[gp_], dEpdC_FourTensor);
      Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(rhs_iFin_V, tempFourTensor);

      // calculate RHS of the equation for the plastic strain
      rhs_epsp_V.update(
          time_step_tracker_.dt_ * state_quantity_derivatives_.curr_dpsr_dequiv_stress_,
          state_quantity_derivatives_.curr_dequiv_stress_dC_, 0.0);
    }
    else if (parameter()->timint_type() == TimIntType::logarithmic)
    // logarithmic substepping
    {
      // calculate RHS of the equation for the plastic deformation gradient
      rhs_iFin_V.update(-time_step_tracker_.dt_, state_quantity_derivatives_.curr_dlpdC_, 0.0);

      // calculate RHS of the equation for the plastic strain
      rhs_epsp_V.update(
          time_step_tracker_.dt_ * state_quantity_derivatives_.curr_dpsr_dequiv_stress_,
          state_quantity_derivatives_.curr_dequiv_stress_dC_, 0.0);
    }
    else
    {
      FOUR_C_THROW("You should not be here");
    }

    // assemble the RHS from its components
    Core::LinAlg::Matrix<10, 6> RHS = assemble_rhs_additional_cmat(rhs_iFin_V, rhs_epsp_V);

    // solve the linear system of equations
    Core::LinAlg::Matrix<10, 6> SOL(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FixedSizeSerialDenseSolver<10, 10, 6> solver;
    solver.set_matrix(jacMat);     // set A = jacM
    solver.set_vectors(SOL, RHS);  // set X=SOL, B=RHS
    solver.factor_with_equilibration(true);
    int err = solver.solve();  // X = A^-1 B
    int err2 = solver.factor();


    if ((err != 0) || (err2 != 0))
    {
      err_status = ErrorType::failed_solution_analytic_linearization;
      evaluate_additional_cmat_perturb_based(FredM, cmatadd, iFin_other, dSdiFinj);

      return;
    }



    // disassemble the solution vector
    Core::LinAlg::Matrix<9, 6> diFinjdCV = extract_derivative_of_inv_inelastic_defgrad(SOL);

    // compute additional term to stiffness matrix additional_cmat
    cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdCV, 1.0);
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // reduced deformation gradient FredM, taking into account all the already computed
  // inelastic factors
  //    \f$ \boldsymbol{F_{\text{red}}} = \boldsymbol{F}
  //    \boldsymbol{F_{\text{in,other}}^{-1}}
  //    \f$
  //      with \f$\boldsymbol{F}_{\text{in,other}}^{-1} = \boldsymbol{F}_{\text{in},1}^{-1}
  //      \boldsymbol{F}_{\text{in},2}^{-1} \dots \f$ up to the current inelastic factor
  Core::LinAlg::Matrix<3, 3> FredM(Core::LinAlg::Initialization::zero);
  FredM.multiply_nn(1.0, *defgrad, iFin_other, 0.0);


  // check whether we have already evaluated the inverse inelastic deformation gradient for
  // the given reduced deformation gradient (this check should only be
  // performed for the first repetition, since we want to repeat the
  // return mapping under the same conditions, and not simply return the
  // computed value)
  Core::LinAlg::Matrix<3, 3> diff_defgrad{Core::LinAlg::Initialization::zero};
  diff_defgrad.update(1.0, FredM, -1.0, time_step_quantities_.current_defgrad_[gp_], 0.0);
  if (diff_defgrad.norm2() <= 1.0e-16)
  {
    // just set the already computed value, no further computation
    iFinM = time_step_quantities_.current_plastic_defgrad_inverse_[gp_];

    return;
  }
  else
  // if this is a "new" deformation gradient, we evaluate the inverse
  // inelastic deformation gradient via return mapping
  {
    if (parameter()->analyze_timint())
    {
      int num_of_required_repetitions = 0;
      Core::LinAlg::Matrix<3, 3> iFinM_temp{Core::LinAlg::Initialization::zero};
      // benchmark run time
      general_local_timint_analysis_utils.time_measurements_.eval_time_rma_ += benchmark_function(
          "Return Mapping", general_local_timint_analysis_utils.timers_.eval_teuchos_timer_rma_,
          parameter()->analyze_timint_timer_rel_tol(),
          general_local_timint_analysis_utils.increment_vars_, num_of_required_repetitions,
          [this](Core::LinAlg::Matrix<3, 3>& FredM, Core::LinAlg::Matrix<3, 3>& iFinM_temp)
          { iFinM_temp = return_mapping(FredM); }, FredM, iFinM_temp);
      iFinM = iFinM_temp;
    }
    else
    {
      iFinM = return_mapping(FredM);
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> Mat::InelasticDefgradTransvIsotropElastViscoplast::return_mapping(
    const Core::LinAlg::Matrix<3, 3>& FredM)
{
  // declare output: inverse inelastic deformation gradient (after return mapping)
  Core::LinAlg::Matrix<3, 3> iFinM{Core::LinAlg::Initialization::zero};

  // compute right CG tensor corresponding to the given deformation gradient
  Core::LinAlg::Matrix<3, 3> CredM(Core::LinAlg::Initialization::zero);
  CredM.multiply_tn(1.0, FredM, FredM, 0.0);

  // perform non-repeatable pre-evaluation tasks (non-repeatable: not
  // called in the redundant evaluate call, which is already handled -> direct return
  // without calling this function)
  prepare_return_mapping(FredM);

  // set predictor: assume purely elastic behavior in this time step
  Core::LinAlg::Matrix<3, 3> iFinM_pred(Core::LinAlg::Initialization::zero);
  iFinM_pred.update(1.0, time_step_quantities_.last_plastic_defgrad_inverse_[gp_], 0.0);
  double plastic_strain_pred = time_step_quantities_.last_plastic_strain_[gp_];

  // set error status of evaluation to no errors
  ErrorType err_status = ErrorType::no_errors;

  // set current defgrad and current right CG tensor
  time_step_quantities_.current_defgrad_[gp_] = FredM;
  time_step_quantities_.current_rightCG_[gp_] = CredM;

  // check whether the predictor is the solution (no plastic strain during this time step)
  bool pred_is_sol = check_elastic_predictor(CredM, iFinM_pred, plastic_strain_pred, err_status);

  if ((err_status == ErrorType::no_errors) && (pred_is_sol))
  {
    // update inverse inelastic defgrad
    iFinM = iFinM_pred;

    // update history variables of material
    if (update_hist_var_)
    {
      time_step_quantities_.current_plastic_defgrad_inverse_[gp_] = iFinM;
      time_step_quantities_.current_plastic_strain_[gp_] = plastic_strain_pred;
      time_step_quantities_.current_equiv_stress_[gp_] = state_quantities_.curr_equiv_stress_;
    }
  }
  else  // predictor does not suffice
  {
    // perform time integration via the Local Newton-Raphson Loop (LNL)
    Core::LinAlg::Matrix<10, 1> x = wrap_unknowns(iFinM_pred, plastic_strain_pred);
    Core::LinAlg::Matrix<10, 1> x_adapted{x};

    // adapt predictor (interpolate Local Newton guess)
    if (parameter()->use_lngi() && !use_elastic_predictor_)
    {
      // increment the number of performed Local Newton Guess Interpolations
      ++lnl_guess_interpolation_.num_of_lngi_;

      x_adapted = interpolate_local_newton_guess(FredM);
    }

    // perform Local Newton Loop (LNL)
    Core::LinAlg::Matrix<10, 1> sol{Core::LinAlg::Initialization::zero};
    sol = local_newton_loop(FredM, x_adapted, err_status);

    // throw error if the Local Newton Loop cannot be evaluated with the given
    // settings
    if (err_status != ErrorType::no_errors)
    {
      // general local time integration analysis output routine with LNL error
      if (parameter()->analyze_timint())
        general_local_timint_analysis_utils.output_error_local_newton_loop(
            local_substepping_utils_.substep_counter_);

      // output error and then throw (in order to display the error on
      // the right processor)
      std::cout << debug_get_error_info(Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
                           get_detailed_error_message_for_error_type(err_status))
                << std::endl;
      FOUR_C_THROW("See above");
    }
    else
    {
      // general local time integration analysis: add number of substeps and stop started
      // timer
      if (parameter()->analyze_timint() && general_local_timint_analysis_utils.increment_vars_)
      {
        // general local time integration analysis: add number of substeps to
        // general_local_timint_analysis_utils
        general_local_timint_analysis_utils.num_iters_and_steps_.eval_num_of_LNL_steps_ += 1;
      }

      // increment number of LNL iterations for the current timestep at the
      // current GP
      lnl_data_.num_iter_curr_timestep_[gp_] += lnl_data_.iter_;
    }


    // extract the inverse inelastic defgrad from the LNL solution
    iFinM = extract_inverse_inelastic_defgrad(sol);

    // update history variables of material
    if (update_hist_var_)
    {
      time_step_quantities_.current_plastic_defgrad_inverse_[gp_] = iFinM;
      time_step_quantities_.current_plastic_strain_[gp_] = sol(9);
      time_step_quantities_.current_equiv_stress_[gp_] = state_quantities_.curr_equiv_stress_;
    }
  }


  return iFinM;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::update()
{
  // reset global iteration tracker
  globiter_ = 0;

  // loop over Gauss points to store relevant data last_ <- current_ (plastic strain incremements,
  // data related to elastic and plastic predictors for LNGI, ...)
  ErrorType err_status = InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;
  for (unsigned int gp = 0; gp < time_step_quantities_.last_rightCG_.size(); ++gp)
  {
    // compute state quantities
    StateQuantities current_state_quantities =
        evaluate_state_quantities(time_step_quantities_.current_rightCG_[gp],
            time_step_quantities_.current_plastic_defgrad_inverse_[gp],
            time_step_quantities_.current_plastic_strain_[gp], err_status, time_step_tracker_.dt_,
            StateQuantityEvalType::PlasticStrainRateOnly);
    FOUR_C_ASSERT_ALWAYS(
        err_status == InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors,
        "Something went wrong while evaluating the current state during update!");

    // --> store plastic strain increment
    time_step_quantities_.last_plastic_strain_increment_[gp] =
        time_step_tracker_.dt_ * current_state_quantities.curr_equiv_plastic_strain_rate_;

    // --> store LNGI predictor data and update last elastic stretches
    if (parameter()->use_lngi())
    {
      if (parameter()->analyze_timint())
      {
        int num_of_required_repetitions = 0;
        // benchmark run time
        general_local_timint_analysis_utils.time_measurements_.eval_time_lngi_preparation_ +=
            benchmark_function(
                "LNGI: Update GP data for next time step",
                general_local_timint_analysis_utils.timers_.eval_teuchos_timer_lngi_preparation_,
                parameter()->analyze_timint_timer_rel_tol(),
                general_local_timint_analysis_utils.increment_vars_, num_of_required_repetitions,
                [this](const unsigned int gp) { update_lngi_data(gp); }, gp);
      }
      else
      {
        update_lngi_data(gp);
      }
    }

    // --> store "main" history variables
    time_step_quantities_.last_rightCG_[gp] = time_step_quantities_.current_rightCG_[gp];
    time_step_quantities_.last_plastic_defgrad_inverse_[gp] =
        time_step_quantities_.current_plastic_defgrad_inverse_[gp];
    time_step_quantities_.last_plastic_strain_[gp] =
        time_step_quantities_.current_plastic_strain_[gp];
    time_step_quantities_.last_equiv_stress_[gp] = time_step_quantities_.current_equiv_stress_[gp];
    time_step_quantities_.last_defgrad_[gp] = time_step_quantities_.current_defgrad_[gp];
  }

  // update history variables for the next time step
  // call update method of the viscoplastic law
  viscoplastic_law_->update();

  // call update method of the LNGI
  if (parameter()->use_lngi())
  {
    if (parameter()->analyze_timint())
    {
      int num_of_required_repetitions = 0;
      // benchmark run time
      general_local_timint_analysis_utils.time_measurements_.eval_time_lngi_preparation_ +=
          benchmark_function("LNGI: Call Update of LNGI Framework",
              general_local_timint_analysis_utils.timers_.eval_teuchos_timer_lngi_preparation_,
              parameter()->analyze_timint_timer_rel_tol(),
              general_local_timint_analysis_utils.increment_vars_, num_of_required_repetitions,
              [this] { lnl_guess_interpolation_.update(); });
    }
    else
    {
      lnl_guess_interpolation_.update();
    }
  }

  // set control variable for the update method of the LNGI; we perform a separate update to avoid
  // expensive computations at the last time instant of the simulation
  compute_lngi_starting_points_ = true;

  // general local time integration analysis: update routine
  if (parameter()->analyze_timint() && general_local_timint_analysis_utils.increment_vars_)
  {
    if (general_local_timint_analysis_utils.num_update_calls_ ==
        Global::Problem::instance()->get_dis("structure")->num_global_elements() - 1)
    {
      // get current interpolation point
      LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
          lnl_guess_interpolation_.get_curr_interp_point(gp_);
      LocalNewtonGuessInterpolation::InterpolationPoint optimal_interp_point =
          lnl_guess_interpolation_.get_optimal_interp_point(gp_);

      // general local time integration analysis: set interpolation factors (the
      // one obtained from the Local Newton Guess Interpolation and the optimal one)
      general_local_timint_analysis_utils.lngi_factors_.curr_lngi_factor_lambda_1_ =
          curr_interp_point.xi_lambda_1_;
      general_local_timint_analysis_utils.lngi_factors_.curr_lngi_factor_lambda_2_ =
          curr_interp_point.xi_lambda_2_;
      general_local_timint_analysis_utils.lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_0_ =
          curr_interp_point.xi_rel_eigenvect_rot_[0];
      general_local_timint_analysis_utils.lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_1_ =
          curr_interp_point.xi_rel_eigenvect_rot_[1];
      general_local_timint_analysis_utils.lngi_factors_.curr_lngi_factor_eigenvect_rot_comp_2_ =
          curr_interp_point.xi_rel_eigenvect_rot_[2];
      general_local_timint_analysis_utils.lngi_factors_.optimal_lngi_factor_lambda_1_ =
          optimal_interp_point.xi_lambda_1_;  // only for the 0-th Gauss point in the
                                              // time integration analysis
      general_local_timint_analysis_utils.lngi_factors_.optimal_lngi_factor_lambda_2_ =
          optimal_interp_point.xi_lambda_2_;  // only for the 0-th Gauss point in the
                                              // time integration analysis
      general_local_timint_analysis_utils.lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_0_ =
          optimal_interp_point.xi_rel_eigenvect_rot_[0];  // only for the 0-th Gauss point in
                                                          // the time integration analysis
      general_local_timint_analysis_utils.lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_1_ =
          optimal_interp_point.xi_rel_eigenvect_rot_[1];  // only for the 0-th Gauss point in
                                                          // the time integration analysis
      general_local_timint_analysis_utils.lngi_factors_.optimal_lngi_factor_eigenvect_rot_comp_2_ =
          optimal_interp_point.xi_rel_eigenvect_rot_[2];  // only for the 0-th Gauss point in
                                                          // the time integration analysis

      // general local time integration analysis: update total values
      general_local_timint_analysis_utils.update_total();

      // general local time integration analysis: write data to csv
      general_local_timint_analysis_utils.write_to_csv();

      // timint_analysis: reset control flow variables
      general_local_timint_analysis_utils.reset_called_ = false;
      general_local_timint_analysis_utils.num_update_calls_ = 0;
    }
    else
    {
      ++general_local_timint_analysis_utils.num_update_calls_;
    }
  }

  // reset number of LNL iterations for current timestep at at all Gauss
  // points
  std::ranges::fill(lnl_data_.num_iter_curr_timestep_, 0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::update_lngi_data(const unsigned int gp)
{
  ErrorType err_status{InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors};

  // compute inverse defgrad
  Core::LinAlg::Matrix<3, 3> inv_defgrad{Core::LinAlg::Initialization::zero};
  inv_defgrad.invert(time_step_quantities_.current_defgrad_[gp]);

  // compute inelastic defgrad within the plastic predictor
  Core::LinAlg::Matrix<3, 3> plastic_pred_inv_inelastic_defgrad =
      lnl_guess_interpolation_.interpolate_inv_plastic_defgrad(gp,
          time_step_quantities_.current_defgrad_[gp],
          LocalNewtonGuessInterpolation::InterpolationPoint{1.0, 1.0, {1.0, 1.0, 1.0}},
          inv_defgrad);

  // compute state quantities related to the elastic and plastic
  // predictors
  StateQuantities elastic_pred_state_quantities =
      evaluate_state_quantities(time_step_quantities_.current_rightCG_[gp],
          time_step_quantities_.last_plastic_defgrad_inverse_[gp],
          time_step_quantities_.last_plastic_strain_[gp], err_status, time_step_tracker_.dt_,
          StateQuantityEvalType::EquivStressOnly);  // only evaluate up to
                                                    // stress
  FOUR_C_ASSERT_ALWAYS(
      err_status == InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors,
      "Something went wrong while evaluating the stress state associated with the "
      "elastic "
      "predictor "
      "during update!");

  StateQuantities plastic_pred_state_quantities = evaluate_state_quantities(
      time_step_quantities_.current_rightCG_[gp], plastic_pred_inv_inelastic_defgrad,
      time_step_quantities_.last_plastic_strain_[gp], err_status, time_step_tracker_.dt_,
      StateQuantityEvalType::EquivStressOnly);  // using the last plastic strain
                                                // should have no influence, because
                                                // we only evaluate the state up to
                                                // the equivalent stress
  FOUR_C_ASSERT_ALWAYS(
      err_status == InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors,
      "Something went wrong while evaluating the state associated with the plastic "
      "predictor "
      "during update!");

  // store predictor data
  time_step_quantities_.last_equiv_stress_elastic_pred_[gp] =
      elastic_pred_state_quantities.curr_equiv_stress_;
  time_step_quantities_.last_equiv_stress_plastic_pred_[gp] =
      plastic_pred_state_quantities.curr_equiv_stress_;


  // --> update the material stretch and the rotation of
  // the inverse inelastic defgrad for the LNGI (last_ values
  // are updated, but we use the current_ values since they were not updated yet)
  Core::LinAlg::Matrix<3, 3> current_plastic_defgrad{Core::LinAlg::Initialization::zero};
  current_plastic_defgrad.invert(time_step_quantities_.current_plastic_defgrad_inverse_[gp]);

  // compute current elastic deformation gradient
  Core::LinAlg::Matrix<3, 3> current_elastic_defgrad{Core::LinAlg::Initialization::zero};
  current_elastic_defgrad.multiply_nn(1.0, time_step_quantities_.current_defgrad_[gp],
      time_step_quantities_.current_plastic_defgrad_inverse_[gp], 0.0);


  // perform its polar-spectral decomposition
  Core::LinAlg::Matrix<3, 3> current_elastic_defgrad_rotation{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> current_elastic_defgrad_material_stretch{
      Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> current_elastic_stretch_eigenval{Core::LinAlg::Initialization::zero};
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> current_elastic_spectral_pairs;

  Core::LinAlg::matrix_3x3_polar_decomposition(current_elastic_defgrad,
      current_elastic_defgrad_rotation, current_elastic_defgrad_material_stretch,
      current_elastic_stretch_eigenval, current_elastic_spectral_pairs);

  // save "last" elastic stretch components
  time_step_quantities_.last_elastic_stretch_eigenval_[gp] = {
      current_elastic_spectral_pairs[0].first, current_elastic_spectral_pairs[1].first,
      current_elastic_spectral_pairs[2].first};
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::determine_lngi_starting_points()
{
  // compute LNGI starting points; we do this only once for all GP currently, since we anyway use
  // only data from the previous time step for the current starting point choices. Might need to be
  // performed each global iteration for more refined starting point choices...
  if (compute_lngi_starting_points_)
  {
    // loop over Gauss points
    for (unsigned int gp = 0; gp < time_step_quantities_.last_plastic_defgrad_inverse_.size(); ++gp)
    {
      // --> compute the optimal interpolation
      // factors for each gp if specified by user
      if (parameter()->lngi_starting_point_type() ==
          InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
              LocalNewtonGuessInterpolationStartingPointType::optimal_interpolation_point)
      {
        // ----------------------------------------------- //
        // first check whether there was plastic flow at all in the previous time step: only then,
        // we calculate the optimal interpolation factors
        // ----------------------------------------------- //

        // verify whether plastic flow occurs: in that case compute the
        // optimal interpolation factors with/without preconditioning;
        // else: set all optimal values to 0.0 = elastic predictor
        if (std::abs(time_step_quantities_.last_plastic_strain_increment_[gp]) > 0.0)
        {
          // get relevant predictor defgrad decomposition (last_, since we determine the optimal
          //  point based on the previous timestep)
          const LocalNewtonGuessInterpolation::PredictorDefgradDecomposition& pred_defgrad_decomp =
              lnl_guess_interpolation_.get_last_pred_decomp_specific_defgrad()[gp];

          // DEBUG
          if (debug_mode(ele_gid_, gp_) && debug_pred_adapt)
          {
            std::cout << "Current predictor defgrad decomposition: " << std::endl;
            const LocalNewtonGuessInterpolation::PredictorDefgradDecomposition&
                debug_curr_pred_defgrad_decomp =
                    lnl_guess_interpolation_.get_curr_pred_decomp_specific_defgrad()[gp];
            debug_curr_pred_defgrad_decomp.print(std::cout);
            std::cout << "Previous predictor defgrad decomposition: " << std::endl;
            pred_defgrad_decomp.print(std::cout);
          }



          if (parameter()->lngi_precondition_matrices())
          {
            lnl_guess_interpolation_.set_optimal_interp_point(
                gp, lnl_guess_interpolation_.compute_optimal_interp_factors(pred_defgrad_decomp, gp,
                        precondition_matrix(time_step_quantities_.last_plastic_defgrad_inverse_[gp],
                            parameter()->lngi_precondition_matrices_num_tol() *
                                time_step_quantities_.last_plastic_defgrad_inverse_[gp].norm2()),
                        precondition_matrix(time_step_quantities_.last_defgrad_[gp],
                            parameter()->lngi_precondition_matrices_num_tol() *
                                time_step_quantities_.last_defgrad_[gp].norm2())));
          }
          else
          {
            lnl_guess_interpolation_.set_optimal_interp_point(
                gp, lnl_guess_interpolation_.compute_optimal_interp_factors(pred_defgrad_decomp, gp,
                        time_step_quantities_.last_plastic_defgrad_inverse_[gp],
                        time_step_quantities_.last_defgrad_[gp]));
          }
        }
        else
        {
          lnl_guess_interpolation_.set_curr_interp_point(
              gp, LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = 0.0,
                      .xi_lambda_2_ = 0.0,
                      .xi_rel_eigenvect_rot_ = {0.0, 0.0, 0.0}});
        }
      }
      else if (parameter()->lngi_starting_point_type() ==
               InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
                   LocalNewtonGuessInterpolationStartingPointType::optimal_equiv_stress)
      {
        // ----------------------------------------------- //
        // first check whether there was plastic flow in the previous time step at all: only then,
        // we calculate the optimal interpolation factors
        // ----------------------------------------------- //

        // verify whether plastic flow occurs: in that case compute the
        // optimal interpolation factors with/without preconditioning;
        // else: set all optimal values to 0.0 = elastic predictor
        if (std::abs(time_step_quantities_.last_plastic_strain_increment_[gp]) > 0.0)
        {
          // stress-based interpolation of the optimal interpolation factors
          double optimal_interp_fact = 0.0;
          const double equiv_stress_plast_pred_min_elast_pred =
              (time_step_quantities_.last_equiv_stress_plastic_pred_[gp] -
                  time_step_quantities_.last_equiv_stress_elastic_pred_[gp]);
          const double equiv_stress_curr_min_elast_pred =
              (time_step_quantities_.last_equiv_stress_[gp] -
                  time_step_quantities_.last_equiv_stress_elastic_pred_[gp]);
          if (std::abs(equiv_stress_plast_pred_min_elast_pred) > 1.0e-8)
          {
            optimal_interp_fact =
                equiv_stress_curr_min_elast_pred / equiv_stress_plast_pred_min_elast_pred;
          }
          lnl_guess_interpolation_.set_optimal_interp_point(gp,
              LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = optimal_interp_fact,
                  .xi_lambda_2_ = optimal_interp_fact,
                  .xi_rel_eigenvect_rot_ = {
                      optimal_interp_fact, optimal_interp_fact, optimal_interp_fact}});
        }
        else
        {
          lnl_guess_interpolation_.set_curr_interp_point(
              gp, LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = 0.0,
                      .xi_lambda_2_ = 0.0,
                      .xi_rel_eigenvect_rot_ = {0.0, 0.0, 0.0}});
        }
      }
    }


    // set control variable for LNGI update to false, to avoid recomputation for now (until the
    // standard update method)
    compute_lngi_starting_points_ = false;
  }

  // set starting point of the Local Newton Guess Interpolation for the current GP
  switch (parameter()->lngi_starting_point_type())
  {
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
        LocalNewtonGuessInterpolationStartingPointType::user_set:
    {
      // get user-set starting point
      const double sp{parameter()->lngi_starting_point()};

      lnl_guess_interpolation_.set_curr_interp_point(
          gp_, LocalNewtonGuessInterpolation::InterpolationPoint{sp, sp, {sp, sp, sp}});
      break;
    }
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
        LocalNewtonGuessInterpolationStartingPointType::last_interpolation_point:
    {
      lnl_guess_interpolation_.set_curr_interp_point(
          gp_, lnl_guess_interpolation_.get_last_interp_point(gp_));
      break;
    }
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
        LocalNewtonGuessInterpolationStartingPointType::optimal_interpolation_point:
    case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
        LocalNewtonGuessInterpolation::LocalNewtonGuessInterpolationStartingPointType::
            optimal_equiv_stress:
    {
      lnl_guess_interpolation_.set_curr_interp_point(
          gp_, lnl_guess_interpolation_.get_optimal_interp_point(gp_));


      break;
    }
    default:
    {
      FOUR_C_THROW("The starting point type {} is not yet supported!",
          parameter()->lngi_starting_point_type());
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::prepare_lngi(
    const Core::LinAlg::Matrix<3, 3>& defgrad)
{
  // numerical tolerance
  const double numerical_tol = 1.0e-10;

  determine_lngi_starting_points();

  // get inverse plastic deformation gradient within the plastic predictor
  Core::LinAlg::Matrix<3, 3> inv_plastic_defgrad_plastic_pred =
      compute_inverse_plastic_defgrad_plastic_pred(defgrad,
          time_step_quantities_.last_plastic_defgrad_inverse_[gp_],
          time_step_quantities_.last_elastic_stretch_eigenval_[gp_],
          parameter()->lngi_plastic_pred_elastic_stretch_eigenval_type(),
          parameter()->lngi_plastic_pred_elastic_stretch_eigenvect_rot_type(),
          parameter()->lngi_plastic_pred_rot_type());


  // consistency check:
  if (parameter()->lngi_check_consistency())
  {
    // compute elastic deformation gradient within the plastic predictor, along with its combined
    // polar and spectral decompositions
    Core::LinAlg::Matrix<3, 3> elastic_defgrad_plast_pred{Core::LinAlg::Initialization::zero};
    elastic_defgrad_plast_pred.multiply_nn(1.0, defgrad, inv_plastic_defgrad_plastic_pred, 0.0);
    Core::LinAlg::Matrix<3, 3> R_elastic_defgrad_plast_pred{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> U_elastic_defgrad_plast_pred{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> lambda_elastic_defgrad_plast_pred{
        Core::LinAlg::Initialization::zero};
    std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>
        spectral_pairs_elastic_defgrad_plast_pred;
    Core::LinAlg::matrix_3x3_polar_decomposition(elastic_defgrad_plast_pred,
        R_elastic_defgrad_plast_pred, U_elastic_defgrad_plast_pred,
        lambda_elastic_defgrad_plast_pred, spectral_pairs_elastic_defgrad_plast_pred);
    Core::LinAlg::Matrix<3, 3> Q_elastic_defgrad_plast_pred{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 1> q{Core::LinAlg::Initialization::zero};
    for (int i = 0; i < 3; ++i)
    {
      q = spectral_pairs_elastic_defgrad_plast_pred[i].second;
      for (int j = 0; j < 3; ++j)
      {
        Q_elastic_defgrad_plast_pred(i, j) = q(j);
      }
    }


    // compute elastic deformation gradient within the elastic predictor, along with its combined
    // polar and spectral decompositions

    Core::LinAlg::Matrix<3, 3> elastic_defgrad_elast_pred{Core::LinAlg::Initialization::zero};
    elastic_defgrad_elast_pred.multiply_nn(
        1.0, defgrad, time_step_quantities_.last_plastic_defgrad_inverse_[gp_], 0.0);
    Core::LinAlg::Matrix<3, 3> R_elastic_defgrad_elast_pred{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> U_elastic_defgrad_elast_pred{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> lambda_elastic_defgrad_elast_pred{
        Core::LinAlg::Initialization::zero};
    std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>
        spectral_pairs_elastic_defgrad_elast_pred;
    Core::LinAlg::matrix_3x3_polar_decomposition(elastic_defgrad_elast_pred,
        R_elastic_defgrad_elast_pred, U_elastic_defgrad_elast_pred,
        lambda_elastic_defgrad_elast_pred, spectral_pairs_elastic_defgrad_elast_pred);
    Core::LinAlg::Matrix<3, 3> Q_elastic_defgrad_elast_pred{Core::LinAlg::Initialization::zero};
    for (int i = 0; i < 3; ++i)
    {
      q = spectral_pairs_elastic_defgrad_elast_pred[i].second;
      for (int j = 0; j < 3; ++j)
      {
        Q_elastic_defgrad_elast_pred(i, j) = q(j);
      }
    }


    if (parameter()->lngi_plastic_pred_elastic_stretch_eigenvect_rot_type() ==
            InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
                PlasticPredictorElasticStretchEigenvectRotType::elastic_predictor &&
        parameter()->lngi_plastic_pred_rot_type() ==
            InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
                PlasticPredictorRotationType::elastic_predictor)
    {
      // reference elastic stretch eigenvalues
      Core::LinAlg::Matrix<3, 3> ref_lambda_elastic_defgrad{Core::LinAlg::Initialization::zero};
      switch (parameter()->lngi_plastic_pred_elastic_stretch_eigenval_type())
      {
        case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
            PlasticPredictorElasticStretchEigenvalType::eliminate:
        {
          for (int i = 0; i < 3; ++i)
          {
            ref_lambda_elastic_defgrad(i, i) = 1.0;
          }
          break;
        }
        case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
            PlasticPredictorElasticStretchEigenvalType::maintain:
        {
          for (int i = 0; i < 3; ++i)
          {
            ref_lambda_elastic_defgrad(i, i) =
                time_step_quantities_.last_elastic_stretch_eigenval_[gp_][i];
          }
          break;
        }
        default:
        {
          FOUR_C_THROW(
              "You should not be here in the consistency check! Incompatible elastic stretch "
              "eigenvalue type {}",
              EnumTools::enum_name(parameter()->lngi_plastic_pred_elastic_stretch_eigenval_type()));
        }
      }
      const double temp_det = ref_lambda_elastic_defgrad.determinant();
      ref_lambda_elastic_defgrad.scale(std::pow(defgrad.determinant() / temp_det, 1.0 / 3.0));


      // reference elastic deformation gradient to check against
      Core::LinAlg::Matrix<3, 3> ref_elastic_defgrad{Core::LinAlg::Initialization::zero};
      Core::LinAlg::Matrix<3, 3> ref_QTLambda{Core::LinAlg::Initialization::zero};
      ref_QTLambda.multiply_tn(1.0, Q_elastic_defgrad_elast_pred, ref_lambda_elastic_defgrad, 0.0);
      Core::LinAlg::Matrix<3, 3> ref_QTLambdaQ{Core::LinAlg::Initialization::zero};
      ref_QTLambdaQ.multiply_nn(1.0, ref_QTLambda, Q_elastic_defgrad_elast_pred, 0.0);
      ref_elastic_defgrad.multiply_nn(1.0, R_elastic_defgrad_elast_pred, ref_QTLambdaQ, 0.0);


      // calculate difference between reference elastic deformation gradient and elastic
      // deformation gradient within the plastic predictor
      Core::LinAlg::Matrix<3, 3> delta_elastic_defgrad{Core::LinAlg::Initialization::zero};
      delta_elastic_defgrad.update(1.0, elastic_defgrad_plast_pred, -1.0, ref_elastic_defgrad, 0.0);

      if (delta_elastic_defgrad.norm2() > numerical_tol)
      {
        std::cout << "Difference of elastic defgrad (plastic predictor) wrt to reference value "
                     "(consistency "
                     "check) is "
                  << delta_elastic_defgrad.norm2() << " > " << numerical_tol
                  << " (numerical tolerance)!" << std::endl;
        std::cout << "plastic_pred: " << std::endl;
        elastic_defgrad_plast_pred.print(std::cout);
        std::cout << "reference: " << std::endl;
        ref_elastic_defgrad.print(std::cout);
        FOUR_C_THROW("Stop");
      }
    }
  }

  // preevaluate Local Newton Guess Interpolation factors for the initial plastic predictor
  if (parameter()->lngi_precondition_matrices())
  {
    lnl_guess_interpolation_.pre_evaluate(gp_,
        precondition_matrix(time_step_quantities_.last_plastic_defgrad_inverse_[gp_],
            parameter()->lngi_precondition_matrices_num_tol() *
                time_step_quantities_.last_plastic_defgrad_inverse_[gp_].norm2()),
        precondition_matrix(
            inv_plastic_defgrad_plastic_pred, parameter()->lngi_precondition_matrices_num_tol() *
                                                  inv_plastic_defgrad_plastic_pred.norm2()),
        precondition_matrix(
            defgrad, parameter()->lngi_precondition_matrices_num_tol() * defgrad.norm2()));
  }
  else
  {
    lnl_guess_interpolation_.pre_evaluate(gp_,
        time_step_quantities_.last_plastic_defgrad_inverse_[gp_], inv_plastic_defgrad_plastic_pred,
        defgrad);
  }

  // consistency check: can we recover the inverse plastic deformation
  // gradient within the plastic predictor from its extract spectral-polar decomposed
  // parts?
  if (parameter()->lngi_check_consistency())
  {
    const LocalNewtonGuessInterpolation::PredictorDefgradDecomposition&
        curr_pred_decomp_specific_defgrad =
            lnl_guess_interpolation_.get_curr_pred_decomp_specific_defgrad()[gp_];

    // compute input matrix from its components
    Core::LinAlg::Matrix<3, 3> recovered_matrix = compute_matrix_from_decomposed_components(
        lnl_guess_interpolation_.get_curr_pred_decomp_specific_defgrad()[gp_].lambda_plast_pred_[0],
        curr_pred_decomp_specific_defgrad.lambda_plast_pred_[1],
        curr_pred_decomp_specific_defgrad.lambda_plast_pred_[2],
        curr_pred_decomp_specific_defgrad.Qmat_elast_pred_,
        curr_pred_decomp_specific_defgrad.Qvec_plast_pred_rel_,
        curr_pred_decomp_specific_defgrad.Rmat_plast_pred_);
    // verify inverse inelastic deformation gradient
    Core::LinAlg::Matrix<3, 3> recovered_inv_plastic_defgrad{Core::LinAlg::Initialization::zero};
    if (lnl_guess_interpolation_.get_defgrad_type() ==
        InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
            DefgradType::elastic_defgrad)
    {
      // inverse defgrad
      Core::LinAlg::Matrix<3, 3> inv_defgrad{Core::LinAlg::Initialization::zero};
      inv_defgrad.invert(defgrad);

      // compute recovered inverse plastic defgrad
      recovered_inv_plastic_defgrad.multiply_nn(1.0, inv_defgrad, recovered_matrix, 0.0);
    }
    else if (lnl_guess_interpolation_.get_defgrad_type() ==
             InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
                 DefgradType::inv_plastic_defgrad)
    {
      recovered_inv_plastic_defgrad = recovered_matrix;
    }
    else
    {
      FOUR_C_THROW("Unsupported deformation gradient type {} for interpolation",
          lnl_guess_interpolation_.get_defgrad_type());
    }


    // compute difference between input matrix and the recovered
    // matrix
    Core::LinAlg::Matrix<3, 3> delta_input_matrix{Core::LinAlg::Initialization::zero};
    delta_input_matrix.update(
        1.0, inv_plastic_defgrad_plastic_pred, -1.0, recovered_inv_plastic_defgrad, 0.0);

    // verify the recovered matrix
    if (delta_input_matrix.norm2() > numerical_tol)
    {
      std::cout << "The determined inverse inelastic deformation gradient within the plastic "
                   "predictor cannot be recovered!"
                << std::endl;
      std::cout << "determined: " << std::endl;
      inv_plastic_defgrad_plastic_pred.print(std::cout);
      std::cout << "recovered: " << std::endl;
      recovered_inv_plastic_defgrad.print(std::cout);
      std::cout << "eigenvalues: " << std::endl;
      std::cout << curr_pred_decomp_specific_defgrad.lambda_plast_pred_[0] << ", "
                << curr_pred_decomp_specific_defgrad.lambda_plast_pred_[1] << ", "
                << curr_pred_decomp_specific_defgrad.lambda_plast_pred_[2] << std::endl;
      std::cout << "eigenvector matrix (elastic predictor): " << std::endl;
      curr_pred_decomp_specific_defgrad.Qmat_elast_pred_.print(std::cout);
      std::cout << "relative eigenvector rotation (wrt elastic predictor): " << std::endl;
      curr_pred_decomp_specific_defgrad.Qvec_plast_pred_rel_.print(std::cout);
      std::cout << "rotation matrix: " << std::endl;
      curr_pred_decomp_specific_defgrad.Rmat_plast_pred_.print(std::cout);
      FOUR_C_THROW("Failed consistency check for Local Newton Guess Interpolation");
    }
  }


  // TODO: routine for determining exact plastic predictor (yield surface);
  // evaluate the states associated with both predictors
  ErrorType err_status{ErrorType::no_errors};
  Core::LinAlg::Matrix<3, 3> right_cg{Core::LinAlg::Initialization::zero};
  right_cg.multiply_tn(1.0, defgrad, defgrad, 0.0);

  StateQuantities state_quantities_plastic_pred = evaluate_state_quantities(right_cg,
      inv_plastic_defgrad_plastic_pred, time_step_quantities_.last_plastic_strain_[gp_], err_status,
      time_step_tracker_.dt_, StateQuantityEvalType::PlasticStrainRateOnly);
  if (err_status != InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
  {
    state_quantities_plastic_pred.curr_equiv_plastic_strain_rate_ = 1.0;
  }



  // compute plastic strain increments for both predictors
  const double plastic_strain_increment_plast_pred = std::abs(
      state_quantities_plastic_pred.curr_equiv_plastic_strain_rate_ * time_step_tracker_.dt_);

  err_status = InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;
  StateQuantities state_quantities_elastic_pred =
      evaluate_state_quantities(right_cg, time_step_quantities_.last_plastic_defgrad_inverse_[gp_],
          time_step_quantities_.last_plastic_strain_[gp_], err_status, time_step_tracker_.dt_,
          StateQuantityEvalType::PlasticStrainRateOnly);
  if (err_status != InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
  {
    state_quantities_elastic_pred.curr_equiv_plastic_strain_rate_ = 1.0;
  }


  const double plastic_strain_increment_elast_pred = std::abs(
      state_quantities_elastic_pred.curr_equiv_plastic_strain_rate_ * time_step_tracker_.dt_);



  // DEBUG
  if (debug_mode(ele_gid_, gp_) && debug_pred_adapt)
  {
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "elastic_defgrad_elast_pred: " << std::endl;
    Core::LinAlg::Matrix<3, 3> elastic_defgrad_elast_pred{Core::LinAlg::Initialization::zero};
    elastic_defgrad_elast_pred.multiply_nn(
        1.0, defgrad, time_step_quantities_.last_plastic_defgrad_inverse_[gp_], 0.0);
    elastic_defgrad_elast_pred.print(std::cout);
    std::cout << "plastic_strain_increment_elast_pred: " << plastic_strain_increment_elast_pred
              << std::endl;
    Core::LinAlg::Matrix<3, 3> elastic_defgrad_plast_pred{Core::LinAlg::Initialization::zero};
    elastic_defgrad_plast_pred.multiply_nn(1.0, defgrad, inv_plastic_defgrad_plastic_pred, 0.0);
    elastic_defgrad_plast_pred.print(std::cout);
    std::cout << "plastic_strain_increment_plast_pred: " << plastic_strain_increment_plast_pred
              << std::endl;
  }


  // based on the plastic strain increment: determine whether to update the plastic predictor, or to
  // directly use the elastic predictor (if its plastic strain increment is already small)
  if (plastic_strain_increment_elast_pred <= ref_plastic_strain_increment)
  {
    use_elastic_predictor_ = true;
    return;
  }
  else
  {
    // determine an updated plastic predictor
    if (plastic_strain_increment_plast_pred <= zero_plastic_strain_increment)
    {
      determine_updated_plastic_predictor_lngi(defgrad);
      use_elastic_predictor_ = false;


      // DEBUG
      if (debug_mode(ele_gid_, gp_) && debug_pred_adapt)
      {
        std::cout << std::string(50, '.') << std::endl;
        std::cout << "...after updated plastic predictor..." << std::endl;
        std::cout << "updated_elastic_defgrad_elast_pred: " << std::endl;
        Core::LinAlg::Matrix<3, 3> elastic_defgrad_elast_pred{Core::LinAlg::Initialization::zero};
        Core::LinAlg::Matrix<3, 3> inv_defgrad{Core::LinAlg::Initialization::zero};
        inv_defgrad.invert(defgrad);
        Core::LinAlg::Matrix<3, 3> updated_inv_plastic_defgrad_elast_pred =
            lnl_guess_interpolation_.interpolate_inv_plastic_defgrad(gp_, defgrad,
                LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = 0.0,
                    .xi_lambda_2_ = 0.0,
                    .xi_rel_eigenvect_rot_ = {0.0, 0.0, 0.0}},
                inv_defgrad);
        elastic_defgrad_elast_pred.multiply_nn(
            1.0, defgrad, updated_inv_plastic_defgrad_elast_pred, 0.0);
        elastic_defgrad_elast_pred.print(std::cout);
        StateQuantities state_quantities_updated_elastic_pred = evaluate_state_quantities(right_cg,
            updated_inv_plastic_defgrad_elast_pred, time_step_quantities_.last_plastic_strain_[gp_],
            err_status, time_step_tracker_.dt_, StateQuantityEvalType::PlasticStrainRateOnly);
        const double plastic_strain_increment_updated_elast_pred =
            std::abs(state_quantities_updated_elastic_pred.curr_equiv_plastic_strain_rate_ *
                     time_step_tracker_.dt_);
        std::cout << "updated_plastic_strain_increment_elast_pred: "
                  << plastic_strain_increment_updated_elast_pred << std::endl;

        std::cout << "updated_elastic_defgrad_plast_pred: " << std::endl;
        Core::LinAlg::Matrix<3, 3> elastic_defgrad_plast_pred{Core::LinAlg::Initialization::zero};
        Core::LinAlg::Matrix<3, 3> updated_inv_plastic_defgrad_plast_pred =
            lnl_guess_interpolation_.interpolate_inv_plastic_defgrad(gp_, defgrad,
                LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = 1.0,
                    .xi_lambda_2_ = 1.0,
                    .xi_rel_eigenvect_rot_ = {1.0, 1.0, 1.0}},
                inv_defgrad);
        elastic_defgrad_plast_pred.multiply_nn(
            1.0, defgrad, updated_inv_plastic_defgrad_plast_pred, 0.0);
        elastic_defgrad_plast_pred.print(std::cout);
        StateQuantities state_quantities_updated_plastic_pred = evaluate_state_quantities(right_cg,
            updated_inv_plastic_defgrad_plast_pred, time_step_quantities_.last_plastic_strain_[gp_],
            err_status, time_step_tracker_.dt_, StateQuantityEvalType::PlasticStrainRateOnly);
        const double plastic_strain_increment_updated_plast_pred =
            std::abs(state_quantities_updated_plastic_pred.curr_equiv_plastic_strain_rate_ *
                     time_step_tracker_.dt_);

        std::cout << "updated_plastic_strain_increment_plast_pred: "
                  << plastic_strain_increment_updated_plast_pred << std::endl;
      }
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::determine_updated_plastic_predictor_lngi(
    const Core::LinAlg::Matrix<3, 3>& defgrad)
{
  // compute right CG tensor
  Core::LinAlg::Matrix<3, 3> CM{Core::LinAlg::Initialization::zero};
  CM.multiply_tn(1.0, defgrad, defgrad, 0.0);

  // declare error status and set to no errors
  ErrorType err_status{ErrorType::no_errors};

  // declare updated plastic predictor for the inverse plastic defgrad and
  // the plastic strain
  Core::LinAlg::Matrix<3, 3> iFin_updated_plastic_pred{Core::LinAlg::Initialization::zero};
  double plastic_strain_updated_plastic_pred{time_step_quantities_.last_plastic_strain_[gp_]};

  // set interval scanning parameter used for plastic predictor update
  const double k_scan_plastic_pred = 0.5;

  // compute inverse defgrad
  Core::LinAlg::Matrix<3, 3> inv_defgrad{Core::LinAlg::Initialization::zero};
  inv_defgrad.invert(defgrad);

  // set interpolation points to be used in the iterative procedure
  LocalNewtonGuessInterpolation::InterpolationPoint lower_bound_interp_point{
      .xi_lambda_1_ = 0.0, .xi_lambda_2_ = 0.0, .xi_rel_eigenvect_rot_ = {0.0, 0.0, 0.0}};
  LocalNewtonGuessInterpolation::InterpolationPoint upper_bound_interp_point{
      .xi_lambda_1_ = 1.0, .xi_lambda_2_ = 1.0, .xi_rel_eigenvect_rot_ = {1.0, 1.0, 1.0}};
  LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point = upper_bound_interp_point;


  // reset the error status to no errors and set the plastic increment to 0.0
  err_status = ErrorType::no_errors;
  double plastic_strain_increment = 0.0;

  // set counter and maximum for iterations needed to get the updated plastic predictor
  unsigned int lngi_plastic_pred_iters = 0;
  const unsigned int max_lngi_plastic_pred_iters = 100;

  // start the procedure of determining the updated upper bound $\xi_u$ for the plastic predictor
  while (true)
  {
    ++lngi_plastic_pred_iters;

    // check whether interpolation is still possible
    if (lngi_plastic_pred_iters > max_lngi_plastic_pred_iters)
    {
      std::cout << debug_get_error_info("Could not determine an updated plastic predictor!")
                << std::endl;
      FOUR_C_THROW("See above");
    }

    // interpolate inverse plastic deformation gradient
    if (is_equal_interp_points(upper_bound_interp_point,
            LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = 0.0,
                .xi_lambda_2_ = 0.0,
                .xi_rel_eigenvect_rot_ = {0.0, 0.0, 0.0}}))
    {
      FOUR_C_THROW(
          "The upper bound in the procedure for updating the plastic predictor is 0.0! Something "
          "went wrong!");
    }
    else
    {
      iFin_updated_plastic_pred = lnl_guess_interpolation_.interpolate_inv_plastic_defgrad(
          gp_, defgrad, curr_interp_point, inv_defgrad);
    }


    // evaluate the current state with the adapted predictor
    StateQuantities state_quantities = evaluate_state_quantities(CM, iFin_updated_plastic_pred,
        plastic_strain_updated_plastic_pred, err_status, time_step_tracker_.dt_,
        StateQuantityEvalType::PlasticStrainRateOnly);

    // check for convergence and adapt used interpolation points based on the obtained error
    if (err_status == ErrorType::no_errors)
    {
      // compute plastic strain increment associated with the computed state
      plastic_strain_increment =
          std::abs(state_quantities.curr_equiv_plastic_strain_rate_ * time_step_tracker_.dt_);

      // check whether we have obtained the relevant interpolation point (yield surface) based on
      // the maximum plastic strain increment
      if (zero_plastic_strain_increment <= plastic_strain_increment)
      {
        if (plastic_strain_increment <= ref_plastic_strain_increment)
        {
          // recall pre-evaluate routine of the LNGI with the updated plastic deformation gradient
          // within the plastic predictor
          if (parameter()->lngi_precondition_matrices())
          {
            lnl_guess_interpolation_.pre_evaluate(gp_,
                precondition_matrix(time_step_quantities_.last_plastic_defgrad_inverse_[gp_],
                    parameter()->lngi_precondition_matrices_num_tol() *
                        time_step_quantities_.last_plastic_defgrad_inverse_[gp_].norm2()),
                precondition_matrix(
                    iFin_updated_plastic_pred, parameter()->lngi_precondition_matrices_num_tol() *
                                                   iFin_updated_plastic_pred.norm2()),
                precondition_matrix(
                    defgrad, parameter()->lngi_precondition_matrices_num_tol() * defgrad.norm2()));
          }
          else
          {
            lnl_guess_interpolation_.pre_evaluate(gp_,
                time_step_quantities_.last_plastic_defgrad_inverse_[gp_], iFin_updated_plastic_pred,
                defgrad);
          }
          // we can now exit the loop
          break;
        }
        else
        {
          lower_bound_interp_point = curr_interp_point;
        }
      }
      else
      {
        upper_bound_interp_point = curr_interp_point;
      }
    }
    else
    {
      // adapt interval bounds based on the error type
      const LocalNewtonGuessInterpolation::PlasticPredUpdateErrorAction err_action =
          LocalNewtonGuessInterpolation::get_plastic_pred_update_error_action(err_status);
      switch (err_action)
      {
        case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
            PlasticPredUpdateErrorAction::shift_towards_plastic_pred:
          lower_bound_interp_point = curr_interp_point;
          break;
        case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonGuessInterpolation::
            PlasticPredUpdateErrorAction::shift_towards_elastic_pred:
          upper_bound_interp_point = curr_interp_point;
          break;
        default:
          FOUR_C_THROW(
              "You should not be here in the plastic predictor update routine! Error action is {}",
              err_action);
      }
    }

    // reset current interpolation point within the updated interpolation interval
    curr_interp_point =
        LocalNewtonGuessInterpolation::add_interpolation_points(1.0 - k_scan_plastic_pred,
            lower_bound_interp_point, k_scan_plastic_pred, upper_bound_interp_point);
  }
  // DEBUG
  if (debug_mode(ele_gid_, gp_) && debug_pred_adapt)
  {
    std::cout << "--> iterations required for updating plastic predictor: "
              << lngi_plastic_pred_iters << std::endl;
  }
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::setup(const int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // auxiliaries
  std::vector<Core::LinAlg::Tensor<double, 3>> temp_vec;
  Core::LinAlg::Matrix<6, 1> temp_6x1(Core::LinAlg::Initialization::zero);

  // default values of the inverse plastic deformation gradient for ALL Gauss Points
  time_step_quantities_.last_plastic_defgrad_inverse_.resize(
      numgp, time_step_quantities_.last_plastic_defgrad_inverse_[0]);
  time_step_quantities_.current_plastic_defgrad_inverse_.resize(numgp,
      time_step_quantities_.last_plastic_defgrad_inverse_[0]);  // value irrelevant at this point
  time_step_quantities_.last_substep_plastic_defgrad_inverse_.resize(
      numgp, time_step_quantities_.last_substep_plastic_defgrad_inverse_[0]);
  time_step_quantities_.last_elastic_stretch_eigenval_.resize(
      numgp, time_step_quantities_.last_elastic_stretch_eigenval_[0]);

  // default values of the plastic strain for ALL Gauss Points
  time_step_quantities_.last_plastic_strain_.resize(
      numgp, time_step_quantities_.last_plastic_strain_[0]);
  time_step_quantities_.current_plastic_strain_.resize(numgp,
      time_step_quantities_.last_plastic_strain_[0]);  // value irrelevant at this point
  time_step_quantities_.last_substep_plastic_strain_.resize(
      numgp, time_step_quantities_.last_substep_plastic_strain_[0]);

  // default value of the plastic strain increments
  time_step_quantities_.last_plastic_strain_increment_.resize(
      numgp, time_step_quantities_.last_plastic_strain_increment_[0]);



  // default values of the equivalent stress for ALL Gauss Points
  time_step_quantities_.last_equiv_stress_.resize(
      numgp, time_step_quantities_.last_equiv_stress_[0]);
  time_step_quantities_.last_equiv_stress_elastic_pred_.resize(
      numgp, time_step_quantities_.last_equiv_stress_elastic_pred_[0]);
  time_step_quantities_.last_equiv_stress_plastic_pred_.resize(
      numgp, time_step_quantities_.last_equiv_stress_plastic_pred_[0]);
  time_step_quantities_.current_equiv_stress_.resize(
      numgp, time_step_quantities_.current_equiv_stress_[0]);  // value irrelevant at this point


  // default values of the right CG deformation tensor for ALL Gauss Points
  time_step_quantities_.last_rightCG_.resize(numgp, time_step_quantities_.last_rightCG_[0]);
  time_step_quantities_.current_rightCG_.resize(
      numgp, time_step_quantities_.last_rightCG_[0]);  // value irrelevant at this point

  // default values of the deformation gradient
  time_step_quantities_.last_defgrad_.resize(numgp, time_step_quantities_.last_defgrad_[0]);
  time_step_quantities_.current_defgrad_.resize(numgp, time_step_quantities_.current_defgrad_[0]);

  // call corresponding method of the viscoplastic law
  viscoplastic_law_->setup(numgp, fibers, coord_system);

  // call setup method of the Local Newton Guess Interpolation
  lnl_guess_interpolation_.setup(numgp);

  // setup the Local Newton data tracker with the correct number
  // of Gauss points
  lnl_data_.set_num_of_gp(numgp);

  // read fiber and structural tensor in the case of transverse isotropy
  if (parameter()->mat_behavior() == MatBehavior::transv_isotrop)
  {
    // read fiber via the fiber reader (hyperelastic transversely isotropic material)
    fiber_reader_.setup(numgp, fibers, coord_system);
    fiber_reader_.get_fiber_vecs(temp_vec);
    m_ = Core::LinAlg::make_matrix<3, 1>(temp_vec.back());
  }
  else
  {
    m_.scale(0.0);
  }
  // set material dependent constant tensors
  const_mat_tensors_.set_material_const_tensors(m_);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::pack_inelastic(
    Core::Communication::PackBuffer& data) const
{
  // pack history variables of this specific inelastic factor
  if (parameter() != nullptr)
  {
    // pack viscoplastic law
    viscoplastic_law_->pack_viscoplastic_law(data);

    // pack interpolation factors
    lnl_guess_interpolation_.pack(data);

    // pack fiber direction
    add_to_pack(data, m_);

    // pack last_ values inside time_step_quantities_
    add_to_pack(data, time_step_quantities_.last_rightCG_);
    add_to_pack(data, time_step_quantities_.last_plastic_defgrad_inverse_);
    add_to_pack(data, time_step_quantities_.last_elastic_stretch_eigenval_);
    add_to_pack(data, time_step_quantities_.last_plastic_strain_);
    add_to_pack(data, time_step_quantities_.last_plastic_strain_increment_);
    add_to_pack(data, time_step_quantities_.last_equiv_stress_);
    add_to_pack(data, time_step_quantities_.last_equiv_stress_elastic_pred_);
    add_to_pack(data, time_step_quantities_.last_equiv_stress_plastic_pred_);
    add_to_pack(data, time_step_quantities_.last_substep_plastic_defgrad_inverse_);
    add_to_pack(data, time_step_quantities_.last_substep_plastic_strain_);
    add_to_pack(data, time_step_quantities_.last_defgrad_);
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::unpack_inelastic(
    Core::Communication::UnpackBuffer& buffer)
{
  // NOTE: factory method is called during assign_to_source in the unpack method of the
  // multiplicative split framework --> material created with its params (as well as the
  // viscoplastic law with its params), we only need to unpack the history variablesj
  if (parameter() != nullptr)
  {
    // unpack viscoplastic law
    viscoplastic_law_->unpack_viscoplastic_law(buffer);

    // unpack interpolation factors
    lnl_guess_interpolation_.unpack(buffer);

    // unpack fiber direction
    extract_from_pack(buffer, m_);

    // unpack last_ values inside time_step_quantities_
    extract_from_pack(buffer, time_step_quantities_.last_rightCG_);
    extract_from_pack(buffer, time_step_quantities_.last_plastic_defgrad_inverse_);
    extract_from_pack(buffer, time_step_quantities_.last_elastic_stretch_eigenval_);
    extract_from_pack(buffer, time_step_quantities_.last_plastic_strain_);
    extract_from_pack(buffer, time_step_quantities_.last_plastic_strain_increment_);
    extract_from_pack(buffer, time_step_quantities_.last_equiv_stress_);
    extract_from_pack(buffer, time_step_quantities_.last_equiv_stress_elastic_pred_);
    extract_from_pack(buffer, time_step_quantities_.last_equiv_stress_plastic_pred_);
    extract_from_pack(buffer, time_step_quantities_.last_substep_plastic_defgrad_inverse_);
    extract_from_pack(buffer, time_step_quantities_.last_substep_plastic_strain_);
    extract_from_pack(buffer, time_step_quantities_.last_defgrad_);
  }

  // fill current_ values with the last_ values
  time_step_quantities_.current_rightCG_.resize(time_step_quantities_.last_rightCG_.size(),
      time_step_quantities_.last_rightCG_[0]);  // value irrelevant
  time_step_quantities_.current_plastic_defgrad_inverse_.resize(
      time_step_quantities_.last_plastic_defgrad_inverse_.size(),
      time_step_quantities_.last_plastic_defgrad_inverse_[0]);  // value irrelevant
  time_step_quantities_.current_plastic_strain_.resize(
      time_step_quantities_.last_plastic_strain_.size(),
      time_step_quantities_.last_plastic_strain_[0]);  // value irrelevant
  time_step_quantities_.current_equiv_stress_.resize(
      time_step_quantities_.last_equiv_stress_.size(),
      time_step_quantities_.last_equiv_stress_[0]);  // value irrelevant

  // set evaluated deformation gradient to 0, to make sure that the inverse inelastic
  // deformation gradient is evaluated fully after the restart
  time_step_quantities_.current_defgrad_.resize(
      time_step_quantities_.last_substep_plastic_defgrad_inverse_.size(),
      Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});


  // set number of Gauss points for the Local Newton data tracker
  lnl_data_.set_num_of_gp(time_step_quantities_.last_plastic_strain_.size());

  // now that the fiber direction is available, we set the material-dependent constant
  // tensors with it
  const_mat_tensors_.set_material_const_tensors(m_);

  // set control variable of the LNGI update
  compute_lngi_starting_points_ = true;

  // set global iteration number to its initial value (as in the setup method)
  globiter_ = -1;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::calculate_local_newton_loop_residual(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<10, 1>& x,
    const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double last_plastic_strain, const double dt,
    ErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // extract inverse inelastic defgrad and plastic strain from input vector
  Core::LinAlg::Matrix<3, 3> iFinM = extract_inverse_inelastic_defgrad(x);
  double plastic_strain = x(9);

  // evaluate state variables
  state_quantities_ = evaluate_state_quantities(
      CM, iFinM, plastic_strain, err_status, dt, StateQuantityEvalType::FullEval);

  // declare residuals of the LNL
  Core::LinAlg::Matrix<3, 3> resFM(Core::LinAlg::Initialization::zero);
  double resepsp = 0.0;

  // compute residuals (standard time integration)
  if (parameter()->timint_type() == TimIntType::standard)
  {
    // calculate residual of the equation for inelastic defgrad
    temp3x3.multiply_nn(1.0, last_iFinM, state_quantities_.curr_EpM_, 0.0);
    resFM.update(1.0, iFinM, -1.0, temp3x3, 0.0);

    // calculate residual of the equation for plastic strain
    resepsp = plastic_strain - last_plastic_strain -
              dt * state_quantities_.curr_equiv_plastic_strain_rate_;
  }
  else if (parameter()->timint_type() == TimIntType::logarithmic)
  // compute residuals (logarithmic substepping)
  {
    // calculate the tensor logarithm involved in the residual
    Core::LinAlg::Matrix<3, 3> last_FinM(Core::LinAlg::Initialization::zero);
    last_FinM.invert(last_iFinM);
    Core::LinAlg::Matrix<3, 3> T(Core::LinAlg::Initialization::zero);
    T.multiply_nn(1.0, last_FinM, iFinM, 0.0);
    Core::LinAlg::MatrixFunctErrorType log_err_status =
        Core::LinAlg::MatrixFunctErrorType::no_errors;
    Core::LinAlg::Matrix<3, 3> logT{Core::LinAlg::Initialization::zero};
    if (parameter()->mat_log_calc_method() ==
        Core::LinAlg::MatrixLogCalcMethod::inv_scal_square)  // evaluation using the inverse
                                                             // scaling and squaring
                                                             // method?...
    {
      // when computing the matrix logarithm with the inverse scaling
      // and squaring, we also save the resulting Pade
      // order via the dedicated pointer. This will be helpful when we
      // compute the derivative - we want consistent Pade orders for the
      // evaluations of functions and their derivatives.
      logT = Core::LinAlg::matrix_log(
          T, log_err_status, matrix_exp_log_utils_.pade_order_, parameter()->mat_log_calc_method());
    }
    else  // evaluation using other provided methods?...
    {
      logT = Core::LinAlg::matrix_log(T, log_err_status, parameter()->mat_log_calc_method());
    }
    if (log_err_status != Core::LinAlg::MatrixFunctErrorType::no_errors)
    {
      err_status = ErrorType::failed_matrix_log_evaluation;
      return Core::LinAlg::Matrix<10, 1>{Core::LinAlg::Initialization::zero};
    }

    // calculate residual of the equation for inelastic defgrad
    resFM.update(1.0, logT, dt, state_quantities_.curr_lpM_, 0.0);

    // calculate residual of the equation for plastic strain
    resepsp = plastic_strain - last_plastic_strain -
              dt * state_quantities_.curr_equiv_plastic_strain_rate_;
  }
  else
  {
    FOUR_C_THROW("You should not be here");
  }

  // return 10x1 residual vector
  Core::LinAlg::Matrix<10, 1> residual;
  residual(0) = resFM(0, 0);
  residual(1) = resFM(1, 1);
  residual(2) = resFM(2, 2);
  residual(3) = resFM(0, 1);
  residual(4) = resFM(1, 2);
  residual(5) = resFM(0, 2);
  residual(6) = resFM(1, 0);
  residual(7) = resFM(2, 1);
  residual(8) = resFM(2, 0);
  residual(9) = resepsp;

  return residual;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 10> Mat::InelasticDefgradTransvIsotropElastViscoplast::calculate_jacobian(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<10, 1>& x,
    const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double last_plastic_strain, const double dt,
    ErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::FourTensor<3> tempFourTensor(true);
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // extract inverse inelastic defgrad and plastic strain from input vector
  Core::LinAlg::Matrix<3, 3> iFinM = extract_inverse_inelastic_defgrad(x);
  double plastic_strain = x(9);

  // evaluate state derivatives
  state_quantity_derivatives_ =
      evaluate_state_quantity_derivatives(CM, iFinM, plastic_strain, err_status, dt,
          StateQuantityDerivEvalType::FullEval);  // we do not reevaluate the state
                                                  // quantities, this was done in the
                                                  // residual computation already

  // get derivative of update tensor wrt inverse inelastic defgrad (in FourTensor form)
  Core::LinAlg::FourTensor<3> dEpdiFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
      dEpdiFin_FourTensor, state_quantity_derivatives_.curr_dEpdiFin_);

  // declare Jacobian component blocks

  // derivative of residual for inelastic deformation gradient w.r.t. inelastic deformation
  // gradient
  Core::LinAlg::Matrix<9, 9> J_iFin_iFin(Core::LinAlg::Initialization::zero);
  // derivative of residual for inelastic deformation gradient w.r.t. plastic strain
  Core::LinAlg::Matrix<9, 1> J_iFin_epsp(Core::LinAlg::Initialization::zero);
  // derivative of residual for plastic strain w.r.t. inelastic deformation gradient
  Core::LinAlg::Matrix<1, 9> J_epsp_iFin(Core::LinAlg::Initialization::zero);
  // derivative of residual for plastic strain w.r.t. plastic strain
  double J_epsp_epsp = 0.0;

  // standard time integration
  if (parameter()->timint_type() == TimIntType::standard)
  {
    // compute 9x9 north-west component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. inelastic deformation gradient)
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        tempFourTensor, last_iFinM, dEpdiFin_FourTensor, true);
    Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, tempFourTensor);
    J_iFin_iFin.update(1.0, const_non_mat_tensors.id4_9x9_, -1.0, temp9x9, 0.0);

    // compute derivative of update tensor wrt plastic strain in matrix form
    Core::LinAlg::Matrix<3, 3> dEpdepsp_M(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_9x1_to_3x3(state_quantity_derivatives_.curr_dEpdepsp_, dEpdepsp_M);

    // compute 9x1 north-east component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. plastic strain)
    temp3x3.multiply_nn(-1.0, last_iFinM, dEpdepsp_M, 0.0);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(temp3x3, J_iFin_epsp);

    // compute 1x9 south-west component block of the Jacobian (derivative of residual for
    // plastic strain w.r.t. inelastic deformation gradient)
    J_epsp_iFin.update(-dt * state_quantity_derivatives_.curr_dpsr_dequiv_stress_,
        state_quantity_derivatives_.curr_dequiv_stress_diFin_, 0.0);

    // compute south-east component of the Jacobian (derivative of residual for plastic
    // strain w.r.t. plastic strain)
    J_epsp_epsp = 1.0 - dt * state_quantity_derivatives_.curr_dpsr_depsp_;
  }
  else if (parameter()->timint_type() == TimIntType::logarithmic)
  // logarithmic time integration
  {
    // compute 9x9 north-west component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. inelastic deformation gradient)
    Core::LinAlg::Matrix<3, 3> last_FinM(Core::LinAlg::Initialization::zero);
    last_FinM.invert(last_iFinM);
    Core::LinAlg::Matrix<3, 3> T(Core::LinAlg::Initialization::zero);
    T.multiply_nn(1.0, last_FinM, iFinM, 0.0);
    Core::LinAlg::MatrixFunctErrorType log_err_status =
        Core::LinAlg::MatrixFunctErrorType::no_errors;
    Core::LinAlg::Matrix<9, 9> dlogTdT{Core::LinAlg::Initialization::zero};
    if ((parameter()->mat_log_deriv_calc_method() ==
            Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::
                pade_part_fract))  // evaluation using the Pade partial fraction
                                   // expansion?...
    {
      // check whether the logarithm was evaluated with the inverse
      // scaling and squaring method, for which we have also determined
      // a suitable Pade order -> if not so, then we throw error, since
      // this is the only implemented case for now!
      FOUR_C_ASSERT_ALWAYS(
          parameter()->mat_log_calc_method() == Core::LinAlg::MatrixLogCalcMethod::inv_scal_square,
          "Combination of logarithm evaluation methods not implemented yet!");

      dlogTdT = Core::LinAlg::matrix_3x3_log_1st_deriv(T, log_err_status,
          matrix_exp_log_utils_.pade_order_, parameter()->mat_log_deriv_calc_method());
    }
    else  // evaluation using other provided methods?...
    {
      dlogTdT = Core::LinAlg::matrix_3x3_log_1st_deriv(
          T, log_err_status, parameter()->mat_log_deriv_calc_method());
    }

    if (log_err_status != Core::LinAlg::MatrixFunctErrorType::no_errors)
    {
      err_status = ErrorType::failed_matrix_log_evaluation;
      return Core::LinAlg::Matrix<10, 10>{Core::LinAlg::Initialization::zero};
    }
    Core::LinAlg::Matrix<9, 9> dTdiFin(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FourTensorOperations::add_non_symmetric_product(
        1.0, last_FinM, const_non_mat_tensors.id3x3_, dTdiFin);
    Core::LinAlg::Matrix<9, 9> dlogTdiFin(Core::LinAlg::Initialization::zero);
    dlogTdiFin.multiply_nn(1.0, dlogTdT, dTdiFin, 0.0);
    J_iFin_iFin.update(1.0, dlogTdiFin, dt, state_quantity_derivatives_.curr_dlpdiFin_, 0.0);

    // compute 9x1 north-east component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. plastic strain)
    J_iFin_epsp.update(dt, state_quantity_derivatives_.curr_dlpdepsp_, 0.0);

    // compute 1x9 south-west component block of the Jacobian (derivative of residual for
    // plastic strain w.r.t. inelastic deformation gradient)
    J_epsp_iFin.update(-dt * state_quantity_derivatives_.curr_dpsr_dequiv_stress_,
        state_quantity_derivatives_.curr_dequiv_stress_diFin_, 0.0);

    // compute south-east component of the Jacobian (derivative of residual for plastic
    // strain w.r.t. plastic strain)
    J_epsp_epsp = 1.0 - dt * state_quantity_derivatives_.curr_dpsr_depsp_;
  }
  else
  {
    FOUR_C_THROW("You should not be here");
  }

  // assemble and return the Jacobian
  return assemble_jacobian_from_components(J_iFin_iFin, J_iFin_epsp, J_epsp_iFin, J_epsp_epsp);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1> Mat::InelasticDefgradTransvIsotropElastViscoplast::local_newton_loop(
    const Core::LinAlg::Matrix<3, 3>& defgrad, const Core::LinAlg::Matrix<10, 1>& x,
    ErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::Matrix<10, 10> temp10x10(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<10, 1> temp10x1(Core::LinAlg::Initialization::zero);

  // reset all iteration data of the LNL -> we track it in this method afterwards
  lnl_data_.reset_all_iteration_data(gp_);

  // calculate right Cauchy-Green deformation tensor
  Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, defgrad, defgrad, 0.0);

  //  declare LNL matrices, vectors
  // Jacobian matrix
  Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
  // increment of the solution variables
  Core::LinAlg::Matrix<10, 1> dx(Core::LinAlg::Initialization::zero);
  // residual of both equations
  Core::LinAlg::Matrix<10, 1> residual(Core::LinAlg::Initialization::zero);
  double residualNorm2(0.0);

  // declare solvers
  Core::LinAlg::FixedSizeSerialDenseSolver<10, 10, 1> solver_10_10_1;

  // define solution vector
  Core::LinAlg::Matrix<10, 1> sol = x;

  // declare current right CG (tensor interpolated later on in each substep)
  Core::LinAlg::Matrix<3, 3> curr_CM(Core::LinAlg::Initialization::zero);

  // reset substep parameters
  local_substepping_utils_.reset();
  local_substepping_utils_.substep_counter_ = 1;
  local_substepping_utils_.curr_dt_ = time_step_tracker_.dt_;
  local_substepping_utils_.total_num_of_substeps_ = 1;

  // set reference matrices for interpolation
  ref_matrices_ = {time_step_quantities_.last_rightCG_[gp_], CM};

  // declare error status for considering a new substep: used to check whether we have
  // halved the time step too many times (false) or if a new substep is possible (true)
  bool new_substep_status = true;

  // initialize gradient of the quadratic residual $\nabla
  // (\boldsymbol{r}^{T}\boldsymbol{r})$
  Core::LinAlg::Matrix<10, 1> gradient_quadratic_residual{Core::LinAlg::Initialization::zero};

  // declare the line search step size \f$ \alpha \f$
  double alpha = 1.0;

  // relative solution increment \f$ \alpha \frac{ \left| \Delta \boldsymbol{s}^{(l)}
  // \right| }{ \boldsymbol{s}^{l+1}}  \f$ (initially zero, because we have set the full
  // increment dx above to 0)
  double rel_sol_incr_norm{0.0};

  // initialize error management action
  ErrorAction err_action{ErrorAction::continue_iteration};

  // initialize tensor interpolation error status
  Core::LinAlg::TensorInterpolationErrorType tensor_interp_err_status =
      Core::LinAlg::TensorInterpolationErrorType::NoErrors;

  // initialize boolean for convergence check
  bool converged{false};

  // substepping procedures
  while (
      local_substepping_utils_.substep_counter_ <= local_substepping_utils_.total_num_of_substeps_)
  {
    // reset iteration counter
    lnl_data_.iter_ = 0;

    // get the right Cauchy-Green tensor of the current substep
    if (parameter()->use_substepping())
    {
      // interpolate right Cauchy-Green tensor if we use local substepping
      curr_CM = tensor_interpolator_.get_interpolated_matrix(ref_matrices_, ref_locs_,
          (local_substepping_utils_.t_ + local_substepping_utils_.curr_dt_) /
              time_step_tracker_.dt_,
          tensor_interp_err_status);
      if (tensor_interp_err_status != Core::LinAlg::TensorInterpolationErrorType::NoErrors)
      {
        err_status = InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::
            failed_right_cg_interpolation;
        return sol;
      }
    }
    else
    {
      // use the previously computed right Cauchy-Green tensor
      curr_CM = ref_matrices_[1];
    }

    // Newton-Raphson scheme for the current substep
    while (true)
    {
      // set error status to no_errors
      err_status = ErrorType::no_errors;

      // increment iteration counter
      ++lnl_data_.iter_;


      // general local time integration analysis: increment iterations
      if (parameter()->analyze_timint() && general_local_timint_analysis_utils.increment_vars_)
        ++general_local_timint_analysis_utils.num_iters_and_steps_.eval_num_of_iters_;

      // compute residual
      residual = calculate_local_newton_loop_residual(curr_CM, sol,
          time_step_quantities_.last_substep_plastic_defgrad_inverse_[gp_],
          time_step_quantities_.last_substep_plastic_strain_[gp_],
          local_substepping_utils_.curr_dt_, err_status);

      // DEBUG
      if (debug_mode(ele_gid_, gp_) && debug_lnl)
      {
        std::cout << "LNL ITER: " << lnl_data_.iter_ << " / " << lnl_data_.max_iter_ << std::endl;
        std::cout << "x: " << std::endl;
        sol.print(std::cout);
        LocalNewtonGuessInterpolation::InterpolationPoint interp_point_lower =
            lnl_guess_interpolation_.get_lower_bound_interp_point(gp_);
        LocalNewtonGuessInterpolation::InterpolationPoint interp_point_upper =
            lnl_guess_interpolation_.get_upper_bound_interp_point(gp_);
        LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
            lnl_guess_interpolation_.get_curr_interp_point(gp_);


        std::cout << "pred: interp_factor xi (lambda_1): " << interp_point_lower.xi_lambda_1_
                  << " <= " << curr_interp_point.xi_lambda_1_
                  << " <= " << interp_point_upper.xi_lambda_1_ << std::endl;
        std::cout << "residual evaluation status:  " << EnumTools::enum_name(err_status)
                  << std::endl;
        if (err_status == InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
        {
          std::cout << "residual: " << residual.norm2() << std::endl;
          std::cout << "increment: " << alpha * dx.norm2() / sol.norm2() << std::endl;
        }
      }


      // based on the residual evaluation: communicate status and values
      // to the LNL data tracker
      if (err_status == InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
      {
        // LNL data: successful evaluation
        lnl_data_.set_iteration_data(
            CSVOutputTrackingData{.ele_gid_ = ele_gid_,
                .gp_ = gp_,
                .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
                .tnp_ = time_step_tracker_.tnp_,
                .globiter_ = globiter_,
                .lnl_iter_ =
                    lnl_data_.iter_ - 1},  // subtract 1 to match the current loop structure
                                           // updating the iteration count at the beginning
            LocalNewtonData::LocalIterDataCollector{
                .iter_status_ = LocalIterationStatus::residual_evaluation_successful,
                .residual_ = residualNorm2,
                .equiv_stress_ = state_quantities_.curr_equiv_stress_,
                .plastic_strain_ = sol(9),
                .interp_param_ = lnl_guess_interpolation_.get_curr_interp_point(gp_).xi_lambda_1_});
      }
      else
      {
        // LNL data: failed evaluation
        lnl_data_.set_iteration_data(
            CSVOutputTrackingData{.ele_gid_ = ele_gid_,
                .gp_ = gp_,
                .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
                .tnp_ = time_step_tracker_.tnp_,
                .globiter_ = globiter_,
                .lnl_iter_ =
                    lnl_data_.iter_ - 1},  // subtract 1 to match the current loop structure
                                           // updating the iteration count at the beginning
            LocalNewtonData::LocalIterDataCollector{
                .iter_status_ = LocalIterationStatus::residual_evaluation_failed,
                .residual_ = -1.0,
                .equiv_stress_ = state_quantities_.curr_equiv_stress_,
                .plastic_strain_ = sol(9),
                .interp_param_ = lnl_guess_interpolation_.get_curr_interp_point(gp_).xi_lambda_1_});
      }

      // no errors: compute relevant 2-norms: residual and increment; also: check for
      // "stuck" Local Newton
      if (err_status == InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
      {
        // 2-norm of the residual
        residualNorm2 = residual.norm2();

        // 2-norm of the solution increment
        rel_sol_incr_norm = alpha * dx.norm2() / sol.norm2();


        // DEBUG
        if (debug_mode(ele_gid_, gp_) && debug_lnl)
        {
          std::cout << "residual: " << residualNorm2 << std::endl;
        }

        // check for "stuck" Local Newton (check only feasible after the first
        // iteration)
        if ((lnl_data_.iter_ > 1) && (dx.norm2() < sol.norm2() * 1.0e-15))
        {
          // only in the case that the residual is verified, we set an
          // error status
          switch (lnl_data_.conv_check_)
          {
            case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
                ResidualOnly:
            case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
                LocalNewtonConvCheck::ResidualAndIncrement:
              if (residualNorm2 > lnl_data_.res_tol_)
              {
                err_status = InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::
                    no_convergence_local_newton;
              }
              break;
            case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
                IncrementOnly:
              // do nothing
              break;
            default:
              FOUR_C_THROW("You should not be here (check: is Local Newton stuck?)");
          }
        }
      }


      // error management
      temp10x1.update(1.0, sol, 0.0);
      err_action = manage_evaluation_error(err_status, sol, curr_CM);
      if (err_action == ErrorAction::return_solution_with_errors)
      {
        // LNL data: nothing do be done anymore, final error
        lnl_data_.set_iteration_data(
            CSVOutputTrackingData{.ele_gid_ = ele_gid_,
                .gp_ = gp_,
                .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
                .tnp_ = time_step_tracker_.tnp_,
                .globiter_ = globiter_,
                .lnl_iter_ =
                    lnl_data_.iter_ - 1},  // subtract 1 to match the current loop structure
                                           // updating the iteration count at the beginning
            LocalNewtonData::LocalIterDataCollector{
                .iter_status_ = LocalIterationStatus::final_error,
                .residual_ = -1.0,
                .equiv_stress_ = state_quantities_.curr_equiv_stress_,
                .plastic_strain_ = sol(9),
                .interp_param_ = lnl_guess_interpolation_.get_curr_interp_point(gp_).xi_lambda_1_});


        // write the data of the failed LNL to csv
        if (parameter()->use_csv_output_failed_local_newton_iter())
          lnl_data_.write_failed_lnl_iteration_data_to_csv(
              CSVOutputTrackingData{.ele_gid_ = ele_gid_,
                  .gp_ = gp_,
                  .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
                  .tnp_ = time_step_tracker_.tnp_,
                  .globiter_ = globiter_,
                  .lnl_iter_ = lnl_data_.iter_ - 1});

        // return bad solution
        return sol;
      }
      else if (err_action == ErrorAction::next_iteration)
      {
        // recompute dx after conducting adjustments to solution vector
        dx.update(1.0, sol, -1.0, temp10x1, 0.0);

        // proceed with next iteration after performing adjustments due
        // to errors
        continue;
      }

      // check convergence
      switch (lnl_data_.conv_check_)
      {
        case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::ResidualOnly:
          converged = (residualNorm2 <= lnl_data_.res_tol_);
          break;
        case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::IncrementOnly:
          converged = (rel_sol_incr_norm <= lnl_data_.incr_tol_);
          break;
        case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
            ResidualAndIncrement:
          converged =
              (residualNorm2 <= lnl_data_.res_tol_ && rel_sol_incr_norm <= lnl_data_.incr_tol_);
          break;
        default:
          FOUR_C_THROW("You should not be here (convergence checking of the Local Newton Loop)");
      }
      if (converged)
      {
        // this means the current substep has converged: we need to update values of the
        // last_substep_ quantities, the time parameter, the substep count and to
        // break out of the loop of the current substep

        // update time parameter and substep count
        local_substepping_utils_.t_ += local_substepping_utils_.curr_dt_;
        local_substepping_utils_.substep_counter_ += 1;

        // update the values of history variables at the last converged state (if we have
        // not reached the last step yet)
        if (local_substepping_utils_.substep_counter_ <=
            local_substepping_utils_.total_num_of_substeps_)
        {
          time_step_quantities_.last_substep_plastic_defgrad_inverse_[gp_] =
              extract_inverse_inelastic_defgrad(sol);
          time_step_quantities_.last_substep_plastic_strain_[gp_] = sol(9);
          // update last substep history variables of the viscoplastic flow rule
          viscoplastic_law_->update_gp_state(gp_);
        }

        // general local time integration analysis actions
        if (parameter()->analyze_timint() && general_local_timint_analysis_utils.increment_vars_)
        {
          // add number of times the step size of the
          // line search algorithm deviates from 1.0 in the last iter
          if (std::abs(alpha - 1.0) > 1.0e-8)
            general_local_timint_analysis_utils.num_iters_and_steps_
                .eval_num_of_alpha_neq_1_last_iter_ += 1;
        }

        // break out of the substep NR loop
        break;
      }

      // check if maximum iteration is reached: if we have halved the time step the maximum
      // number of times, throw error and finish execution. Otherwise throw exception and
      // proceed with a smaller time step in the substepping scheme!
      if (lnl_data_.iter_ > lnl_data_.max_iter_)
      {
        switch (lnl_data_.diver_cont_)
        {
          case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonDiverCont::
              Stop:
          {
            // substepping procedure
            if (parameter()->use_substepping())
            {
              new_substep_status = prepare_new_substep(sol, curr_CM);
              // if the halving number was exceeded --> return with error
              if (!new_substep_status)
              {
                err_status = ErrorType::no_convergence_local_newton;
                return sol;  // return with error
              }
              continue;
            }


            // write the data of the failed LNL to csv
            if (parameter()->use_csv_output_failed_local_newton_iter())
              lnl_data_.write_failed_lnl_iteration_data_to_csv(
                  CSVOutputTrackingData{.ele_gid_ = ele_gid_,
                      .gp_ = gp_,
                      .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
                      .tnp_ = time_step_tracker_.tnp_,
                      .globiter_ = globiter_,
                      .lnl_iter_ = lnl_data_.iter_ - 1});


            // if no substepping is applied: then we have nor converged,
            // return with error
            err_status = ErrorType::no_convergence_local_newton;
            return sol;

            break;
          }
          case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonDiverCont::
              Continue:
          case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonDiverCont::
              ContinueWithSafeGuard:
          {
            // write the data of the failed LNL to csv
            if (parameter()->use_csv_output_failed_local_newton_iter())
              lnl_data_.write_failed_lnl_iteration_data_to_csv(
                  CSVOutputTrackingData{.ele_gid_ = ele_gid_,
                      .gp_ = gp_,
                      .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
                      .tnp_ = time_step_tracker_.tnp_,
                      .globiter_ = globiter_,
                      .lnl_iter_ = lnl_data_.iter_ - 1});


            // throw warning
            std::cout << "WARNING: The Local Newton Loop for ele_gid = " << ele_gid_
                      << ", gp = " << gp_ << " did not reach convergence after "
                      << lnl_data_.max_iter_ << " iterations: residualNorm2 = " << residualNorm2
                      << ", increment = " << rel_sol_incr_norm << std::endl;

            // safeguard check: is the current solution within the
            // bounds posed by the
            // maximum exceedance?
            if (lnl_data_.diver_cont_ == InelasticDefgradTransvIsotropElastViscoplastUtils::
                                             LocalNewtonDiverCont::ContinueWithSafeGuard)
            {
              switch (lnl_data_.conv_check_)
              {
                case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
                    LocalNewtonConvCheck::ResidualOnly:
                {
                  FOUR_C_ASSERT_ALWAYS(
                      residualNorm2 < (lnl_data_.res_tol_ * lnl_data_.max_exceedance_fact_res_tol_),
                      "Residual {} exceeds the residual tolerance {} by more than the set "
                      "exceedance tolerance factor {}!",
                      residualNorm2, lnl_data_.res_tol_, lnl_data_.max_exceedance_fact_res_tol_);

                  break;
                }
                case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
                    LocalNewtonConvCheck::IncrementOnly:
                {
                  FOUR_C_ASSERT_ALWAYS(
                      rel_sol_incr_norm <
                          (lnl_data_.incr_tol_ * lnl_data_.max_exceedance_fact_incr_tol_),
                      "Relative increment {} exceeds the increment tolerance {} by more "
                      "than the "
                      "set "
                      "exceedance tolerance factor {}!",
                      rel_sol_incr_norm, lnl_data_.incr_tol_,
                      lnl_data_.max_exceedance_fact_incr_tol_);

                  break;
                }
                case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
                    LocalNewtonConvCheck::ResidualAndIncrement:
                {
                  FOUR_C_ASSERT_ALWAYS(
                      (residualNorm2 <
                          (lnl_data_.res_tol_ * lnl_data_.max_exceedance_fact_res_tol_)) &&
                          (rel_sol_incr_norm < (lnl_data_.incr_tol_)),
                      "Residual {} and relative increment {} exceeds the tolerances {} and "
                      "{} by "
                      "more than the "
                      "set "
                      "exceedance tolerance factors {} and {}!",
                      residualNorm2, rel_sol_incr_norm, lnl_data_.res_tol_, lnl_data_.incr_tol_,
                      lnl_data_.max_exceedance_fact_res_tol_, 0.0);


                  break;
                }
                default:
                  FOUR_C_THROW(
                      "You should not be here (safeguard checking for divergence "
                      "management in the "
                      "Local Newton Loop)");
              }
            }

            // we set the error status to no errors (to continue with
            // the simulation even though no convergence was reached)
            err_status = ErrorType::no_errors;
            return sol;

            break;
          }
          default:
            FOUR_C_THROW(
                "You should not be here (divergence management strategy for Local Newton "
                "Loop)");
        }
      }

      // compute Jacobian
      jacMat = calculate_jacobian(curr_CM, sol,
          time_step_quantities_.last_substep_plastic_defgrad_inverse_[gp_],
          time_step_quantities_.last_substep_plastic_strain_[gp_],
          local_substepping_utils_.curr_dt_, err_status);
      // error management
      err_action = manage_evaluation_error(err_status, sol, curr_CM);
      if (err_action == ErrorAction::return_solution_with_errors)
        return sol;
      else if (err_action == ErrorAction::next_iteration)
        continue;

      // set temp variables to residual and jacobian: the original variables will be changed
      // during the solution, and we want to keep them unchanged in order to provide them to
      // the line search algorithm
      temp10x1 = residual;
      temp10x1.scale(-1.0);
      temp10x10 = jacMat;

      // solve loop equation
      dx.clear();                                      // reset
      solver_10_10_1.set_matrix(jacMat);               // set A=jacMat
      solver_10_10_1.set_vectors(dx, temp10x1);        // set dx=increment, residual=RHS
      solver_10_10_1.factor_with_equilibration(true);  // "some easy type of preconditioning"
      int err2 = solver_10_10_1.factor();              // factoring
      int err = solver_10_10_1.solve();                // X = A^-1 B
      if ((err != 0) || (err2 != 0))
      {
        err_status = ErrorType::failed_solution_linear_system_lnl;
        // error management
        err_action = manage_evaluation_error(err_status, sol, curr_CM);
        if (err_action == ErrorAction::return_solution_with_errors)
          return sol;
        else if (err_action == ErrorAction::next_iteration)
          continue;
      }


      // correct Newton direction if it is not a descent direction: use
      // steepest descent direction
      if (parameter()->use_steepest_descent_update_correction())
      {
        // compute gradient of the quadratic residual $\nabla
        // (\boldsymbol{r}^{T}\boldsymbol{r})$
        gradient_quadratic_residual.multiply_tn(2.0, jacMat, residual,
            0.0);  // minus required since we made the residual negative before

        // compute scalar product between the gradient of the
        // quadratic residual and the update direction
        Core::LinAlg::Matrix<1, 1> scalar_product{Core::LinAlg::Initialization::zero};
        scalar_product.multiply_tn(1.0, gradient_quadratic_residual, dx, 0.0);

        // correct Newton direction: steepest descent direction
        if (scalar_product(0) > 0.0)
        {
          // save current norm
          const double norm_dx = dx.norm2();

          // redirect the update vector with the same norm
          dx = gradient_quadratic_residual;
          dx.scale(-norm_dx / gradient_quadratic_residual.norm2());
        }
      }

      // backtracking line search
      if (parameter()->use_line_search())
      {
        // check angle condition
        if (parameter()->check_line_search_angle_condition())
        {
          if (!parameter()->use_steepest_descent_update_correction())
          {
            // compute gradient of the quadratic residual $\nabla
            // (\boldsymbol{r}^{T}\boldsymbol{r})$
            gradient_quadratic_residual.multiply_tn(2.0, jacMat, residual,
                0.0);  // minus required since we made the residual negative before
          }

          // compute scalar product between the gradient of the
          // quadratic residual and the update direction
          Core::LinAlg::Matrix<1, 1> scalar_product{Core::LinAlg::Initialization::zero};
          scalar_product.multiply_tn(1.0, gradient_quadratic_residual, dx, 0.0);

          // compute the tolerance of the scalar product
          const double tol_scalar_product = -parameter()->line_search_angle_condition_tolerance() *
                                            gradient_quadratic_residual.norm2() * dx.norm2();

          // check condition
          if (scalar_product(0) > tol_scalar_product)
          {
            std::cout << "WARNING: violation of line search angle condition in iter "
                      << lnl_data_.iter_ << ": scalar_product=" << scalar_product(0)
                      << " > tol_scalar_product=" << tol_scalar_product << std::endl;
          }
        }


        // compute line search step size
        alpha = get_line_search_step(sol, curr_CM, residual, dx, err_status);

      }  // otherwise it is the default value alpha = 1

      if (err_status == ErrorType::no_errors)
      {
        // update solution vector
        sol.update(alpha, dx, 1.0);
      }
      else
      {
        // try to manage the evaluation error if possible (important especially for failed
        // line search parameter computations)
        err_action = manage_evaluation_error(err_status, sol, curr_CM);
        if (err_action == ErrorAction::return_solution_with_errors)
          return sol;
        else if (err_action == ErrorAction::next_iteration)
          continue;
      }
    }
  }

  // append LNL data (for the successful last iteration)
  lnl_data_.set_iteration_data(
      CSVOutputTrackingData{.ele_gid_ = ele_gid_,
          .gp_ = gp_,
          .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
          .tnp_ = time_step_tracker_.tnp_,
          .globiter_ = globiter_,
          .lnl_iter_ = lnl_data_.iter_ - 1},  // subtract 1 to match the current loop structure
                                              // updating the iteration count at the beginning
      LocalNewtonData::LocalIterDataCollector{.iter_status_ = LocalIterationStatus::converged,
          .residual_ = -1.0,
          .equiv_stress_ = state_quantities_.curr_equiv_stress_,
          .plastic_strain_ = sol(9),
          .interp_param_ = lnl_guess_interpolation_.get_curr_interp_point(gp_).xi_lambda_1_});



  // return the obtained solution
  return sol;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplast::check_elastic_predictor(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<3, 3>& iFinM_pred,
    const double plastic_strain_pred, ErrorType& err_status)
{
  // evaluate state with this elastic predictor and the minimum possible time step
  state_quantities_ = evaluate_state_quantities(CM, iFinM_pred, plastic_strain_pred, err_status,
      time_step_tracker_.min_dt_, StateQuantityEvalType::PlasticStrainRateOnly);


  // DEBUG
  if (debug_mode(ele_gid_, gp_) && debug_lnl)
  {
    std::cout << "Checking elastic predictor: " << std::endl;
    std::cout << "CM: " << std::endl;
    CM.print(std::cout);
    std::cout << "iFinM: " << std::endl;
    iFinM_pred.print(std::cout);
    std::cout << "plastic_strain: " << std::endl;
    std::cout << plastic_strain_pred << std::endl;
    std::cout << "--> equiv_stress: " << state_quantities_.curr_equiv_stress_
              << "; plastic_strain_rate: " << state_quantities_.curr_equiv_plastic_strain_rate_
              << "; elastic predictor = sol: "
              << std::to_string((state_quantities_.curr_equiv_plastic_strain_rate_ < 1.0e-15))
              << std::endl;
  }



  // check if the predicted plastic strain rate is 0 -> for flow rules with yield functions,
  // this means that the predictor is correct
  return (state_quantities_.curr_equiv_plastic_strain_rate_ * time_step_tracker_.dt_ <
          zero_plastic_strain_increment);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplast::prepare_new_substep(
    Core::LinAlg::Matrix<10, 1>& sol, Core::LinAlg::Matrix<3, 3>& curr_CM)
{
  // extract substep parameters
  const double& t = local_substepping_utils_.t_;
  const unsigned int& substep_counter = local_substepping_utils_.substep_counter_;
  double& curr_dt = local_substepping_utils_.curr_dt_;
  unsigned int& time_step_halving_counter = local_substepping_utils_.time_step_halving_counter_;
  unsigned int& total_num_of_substeps = local_substepping_utils_.total_num_of_substeps_;
  unsigned int& iter = lnl_data_.iter_;

  // the current iteration vector has reached a numerically inevaluable state -> we halve
  // the time step and apply substepping

  // halve the current time step
  curr_dt *= 1.0 / 2.0;
  time_step_halving_counter += 1;
  total_num_of_substeps += (total_num_of_substeps - substep_counter + 1);

  // check if we have halved the time step too many times
  if (time_step_halving_counter > parameter()->max_halve_number())
  {
    return false;
  }

  // reset the predictor to the last converged state
  sol = wrap_unknowns(time_step_quantities_.last_substep_plastic_defgrad_inverse_[gp_],
      time_step_quantities_.last_substep_plastic_strain_[gp_]);

  // declare tensor interpolator error status
  Core::LinAlg::TensorInterpolationErrorType tensor_interp_err_status{
      Core::LinAlg::TensorInterpolationErrorType::NoErrors};

  // recompute the current right CG
  curr_CM = tensor_interpolator_.get_interpolated_matrix(
      ref_matrices_, ref_locs_, (t + curr_dt) / time_step_tracker_.dt_, tensor_interp_err_status);
  if (tensor_interp_err_status != Core::LinAlg::TensorInterpolationErrorType::NoErrors)
  {
    std::cout << debug_get_error_info(Core::LinAlg::make_error_message(tensor_interp_err_status))
              << std::endl;
    FOUR_C_THROW("See above");
  }



  // reset iteration counter to 0, as we restart the Newton-Raphson Loop
  iter = 0;

  return true;  // no error
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_additional_cmat_perturb_based(
    const Core::LinAlg::Matrix<3, 3>& FredM, Core::LinAlg::Matrix<6, 6>& cmatadd,
    const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<6, 9>& dSdiFinj)
{
  // ----- FD-based linearization ----- //
  // approximation using perturbations of the right Cauchy-Green deformation tensor,
  // inspired by the procedure described in Miehe et al. (1995)

  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // set update boolean to false
  update_hist_var_ = false;  // no update of the current_ values during the upcoming evaluation
  // of perturbed states

  // inverse of the reduced deformation gradient
  Core::LinAlg::Matrix<3, 3> iFredM(Core::LinAlg::Initialization::zero);
  iFredM.invert(FredM);

  // Voigt representation of the inverse inelastic defgrad
  Core::LinAlg::Matrix<9, 1> iFinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(
      time_step_quantities_.current_plastic_defgrad_inverse_[gp_], iFinV);

  // derivative of inverse inelastic deformation gradient w.r.t. right Cauchy-Green
  // deformation tensor, to be evaluated in the FD-based procedure
  Core::LinAlg::Matrix<9, 6> diFindC_FD(Core::LinAlg::Initialization::zero);

  // declare perturbed variables
  Core::LinAlg::Matrix<3, 3> perturbed_FM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3>* pointer_perturbed_FM = &perturbed_FM;
  Core::LinAlg::Matrix<3, 3> perturbed_CM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> perturbed_iFinM(Core::LinAlg::Initialization::zero);

  // define the delta perturbed deformation gradients
  std::vector<Core::LinAlg::Matrix<3, 3>> delta_perturbed_defgrads(6);
  const double pert_fact = 1.0e-10;  // perturbation factor \f$ \epsilon \f$
  std::vector<std::tuple<int, int>> indices_array = {
      {0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {0, 2}};

  // vary deformation gradient (and therefore the right Cauchy-Green tensor), calculate
  // resulting inverse inelastic defgrad, and compute the contribution to the required
  // derivative
  for (int i = 0; i < static_cast<int>(indices_array.size()); i++)
  {
    // set perturbation of the form
    // \f$ \Delta F_{pert(CD)} = \epsilon/2 F^{-T} (E_{C} \otimes  E_{D} + E_{D} \otimes
    // E_{C} ) \f$
    temp3x3.clear();
    temp3x3(std::get<0>(indices_array[i]), std::get<1>(indices_array[i])) += pert_fact / 2.0;
    temp3x3(std::get<1>(indices_array[i]), std::get<0>(indices_array[i])) += pert_fact / 2.0;
    delta_perturbed_defgrads[i].multiply_tn(1.0, iFredM, temp3x3, 0.0);

    // get perturbed defgrad
    perturbed_FM.update(1.0, FredM, 1.0, delta_perturbed_defgrads[i], 0.0);

    // calculate perturbed right CG tensor
    perturbed_CM.multiply_tn(1.0, perturbed_FM, perturbed_FM, 0.0);

    // get corresponding inverse inelastic defgrad
    evaluate_inverse_inelastic_def_grad(pointer_perturbed_FM, iFin_other, perturbed_iFinM);
    Core::LinAlg::Matrix<9, 1> perturbed_iFinV(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(perturbed_iFinM, perturbed_iFinV);

    // update components of the required derivative
    for (int j = 0; j < 9; ++j)
    {
      diFindC_FD(j, i) += 1.0 / 2.0 * 1.0 / pert_fact * (perturbed_iFinV(j, 0) - iFinV(j, 0));
    }
  }


  // compute additional term to stiffness matrix additional_cmat
  cmatadd.multiply_nn(2.0, dSdiFinj, diFindC_FD, 1.0);

  // reset boolean for the history update
  update_hist_var_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::interpolate_local_newton_guess(
    const Core::LinAlg::Matrix<3, 3>& FM)
{
  // DEBUG
  if (debug_mode(ele_gid_, gp_) && debug_pred_adapt)
  {
    std::cout << "LNGI num " << lnl_guess_interpolation_.num_of_lngi_ << " for ele_gid_ "
              << ele_gid_ << " and gp " << gp_ << std::endl;
  }

  // csv runtime output
  if (parameter()->use_csv_output_lngi_micro_iter())
  {
    // initialize micro iteration data for all microiterations
    // of the subsequent Local Newton Guess Interpolation, to be written to csv
    csv_output_lngi_micro_iter_data_ =
        CSVOutputPredAdaptMicroIterData{CSVOutputTrackingData{.ele_gid_ = ele_gid_,
            .gp_ = gp_,
            .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
            .tnp_ = time_step_tracker_.tnp_,
            .globiter_ = globiter_,
            .lnl_iter_ = lnl_data_.iter_}};
  }


  // compute right CG tensor
  Core::LinAlg::Matrix<3, 3> CM{Core::LinAlg::Initialization::zero};
  CM.multiply_tn(1.0, FM, FM, 0.0);

  // declare error status and set to no errors
  ErrorType err_status{ErrorType::no_errors};

  // declare adapted predictor for the inverse plastic defgrad and
  // the plastic strain
  Core::LinAlg::Matrix<3, 3> iFin_adapt_pred{Core::LinAlg::Initialization::zero};
  double plastic_strain_adapt_pred{0.0};

  // set maximum number of Local Newton Guess Interpolation steps and specific
  // counter
  unsigned int lngi_step_counter = 0;


  // compute inverse defgrad
  Core::LinAlg::Matrix<3, 3> inv_FM{Core::LinAlg::Initialization::zero};
  inv_FM.invert(FM);


  // reset the error status to no errors and set the current equivalent
  // plastic strain rate to 0.0, and start the procedure of determining
  // the interpolation factor $\xi$
  err_status = ErrorType::no_errors;
  state_quantities_.curr_equiv_plastic_strain_rate_ = 0.0;
  while (state_quantities_.curr_equiv_plastic_strain_rate_ * time_step_tracker_.dt_ <= 0.0 ||
         err_status != ErrorType::no_errors)
  {
    ++lngi_step_counter;

    // check whether interpolation is still possible
    if (!lnl_guess_interpolation_.is_interpolation_possible(gp_, lngi_step_counter))
    {
      // write micro iteration data to csv
      if (parameter()->use_csv_output_lngi_micro_iter())
        csv_output_lngi_micro_iter_data_.write_lngi_micro_iter_data_to_csv();

      std::cout << debug_get_error_info("Could not determine an initial guess!") << std::endl;
      FOUR_C_THROW("See above");
    }

    // interpolate inverse plastic deformation gradient
    if (is_equal_interp_points(lnl_guess_interpolation_.get_curr_interp_point(gp_),
            LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = 0.0,
                .xi_lambda_2_ = 0.0,
                .xi_rel_eigenvect_rot_ = {0.0, 0.0, 0.0}}))
    {  // if the interpolation factors are near 0, then we explicitly use the elastic
       // predictor based on the previous timestep; otherwise we will get mismatches due to
       // the interpolation of the elastic predictor which, even though they are within
       // machine precision, may lead to strange effects (such as in stress relaxation:
       // stress of the elastic predictor is >= yield stress, but after interpolation stress
       // < yield stress, which halts the LNGI completely)
      iFin_adapt_pred = time_step_quantities_.last_plastic_defgrad_inverse_[gp_];
    }
    else
    {
      iFin_adapt_pred = lnl_guess_interpolation_.interpolate_inv_plastic_defgrad(
          gp_, FM, lnl_guess_interpolation_.get_curr_interp_point(gp_), inv_FM);
    }

    // compute equivalent stress related to the interpolated inverse plastic deformation
    // gradient
    StateQuantities state_quantities_stress = evaluate_state_quantities(CM, iFin_adapt_pred, 0.0,
        err_status, time_step_tracker_.dt_, StateQuantityEvalType::EquivStressOnly);

    // DEBUG
    if (debug_mode(ele_gid_, gp_) && debug_pred_adapt)
    {
      std::cout << "...integrating plastic strain..." << std::endl;
    }

    // solve for the updated plastic strain (integrate evolution
    // equation with the adapted plastic deformation gradient)
    plastic_strain_adapt_pred = integrate_plastic_strain(state_quantities_stress.curr_equiv_stress_,
        time_step_quantities_.last_plastic_strain_[gp_], time_step_tracker_.dt_, err_status);
    if (err_status == ErrorType::no_errors)
    {
      // evaluate state quantities and their derivatives with interpolated plastic defgrad
      // and integrated plastic strain as initial guess
      is_valid_local_newton_initial_guess(FM, inv_FM, CM, iFin_adapt_pred,
          plastic_strain_adapt_pred, err_status, state_quantities_, state_quantity_derivatives_);
    }

    // DEBUG
    if (debug_mode(ele_gid_, gp_) && debug_pred_adapt)
    {
      std::cout << "ITER:  " << lngi_step_counter << std::endl;
      std::cout << "curr_interp_point: " << std::endl;
      lnl_guess_interpolation_.get_curr_interp_point(gp_).print(std::cout);
      std::cout << "lower_bound_interp_point: " << std::endl;
      lnl_guess_interpolation_.get_lower_bound_interp_point(gp_).print(std::cout);
      std::cout << "upper_bound_interp_point: " << std::endl;
      lnl_guess_interpolation_.get_upper_bound_interp_point(gp_).print(std::cout);
      std::cout << "equiv_stress:  " << state_quantities_stress.curr_equiv_stress_
                << ", plastic_strain: " << plastic_strain_adapt_pred
                << ", plastic_strain_rate: " << state_quantities_.curr_equiv_plastic_strain_rate_
                << std::endl;
      std::cout << "err_status: " << EnumTools::enum_name(err_status) << std::endl;
    }



    // if there was an evaluation error: adapt interpolation interval
    // and the interpolation parameters subsequently
    if (err_status != ErrorType::no_errors)
    {
      // set micro iteration data for the current evaluation
      if (parameter()->use_csv_output_lngi_micro_iter())
      {
        const LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
            lnl_guess_interpolation_.get_curr_interp_point(gp_);
        csv_output_lngi_micro_iter_data_.append_micro_iter_data(
            {
                .current_xi_lambda_1_ = curr_interp_point.xi_lambda_1_,
                .current_xi_lambda_2_ = curr_interp_point.xi_lambda_2_,
                .current_xi_eigenvect_rot_ = curr_interp_point.xi_rel_eigenvect_rot_,
                .current_equiv_stress_ = state_quantities_.curr_equiv_stress_,
                .current_plastic_strain_ = plastic_strain_adapt_pred,
                .current_error_status_ = err_status,
            },
            lngi_step_counter - 1);
      }


      // adapt interpolation interval
      lnl_guess_interpolation_.adapt_interpolation_intervals(gp_, err_status);

      // if the adapted interval is now [0, 0], i.e., the elastic predictor with integrated
      // plastic strain is "under the yield surface": try evaluating the standard elastic
      // predictor as a last resort
      if (is_equal_interp_points(lnl_guess_interpolation_.get_upper_bound_interp_point(gp_),
              LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = 0.0,
                  .xi_lambda_2_ = 0.0,
                  .xi_rel_eigenvect_rot_ = {0.0, 0.0, 0.0}}))
      {
        // set number of lngi to maximum, since LNGI should not be used anymore
        lnl_guess_interpolation_.num_of_lngi_ = lnl_guess_interpolation_.max_num_lngi_;

        // set and return standard elastic predictor
        lnl_guess_interpolation_.guess_inv_plast_defgrad_ =
            wrap_unknowns(time_step_quantities_.last_plastic_defgrad_inverse_[gp_],
                time_step_quantities_.last_plastic_strain_[gp_]);
        return lnl_guess_interpolation_.guess_inv_plast_defgrad_;
      }

      // adapt interpolation parameters
      lnl_guess_interpolation_.adapt_interpolation_parameters(gp_);
    }
  }

  // wrap adapted predictor
  lnl_guess_interpolation_.guess_inv_plast_defgrad_ =
      wrap_unknowns(iFin_adapt_pred, plastic_strain_adapt_pred);

  // general local time integration analysis actions
  if (parameter()->analyze_timint() && general_local_timint_analysis_utils.increment_vars_)
  {
    // general local time integration analysis: save number of performed iterations
    general_local_timint_analysis_utils.num_iters_and_steps_.eval_num_of_lngi_iters_ +=
        lngi_step_counter;

    // general local time integration analysis: perform the same actions for
    // reinterpolation, if this is the case
    if (lnl_guess_interpolation_.num_of_lngi_ >= 1)
    {
      general_local_timint_analysis_utils.num_iters_and_steps_.eval_num_of_reinterp_iters_ +=
          lngi_step_counter;
    }
  }
  // append micro iteration data for the last micro iteration which
  // was successful
  if (parameter()->use_csv_output_lngi_micro_iter())
  {
    const LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
        lnl_guess_interpolation_.get_curr_interp_point(gp_);
    csv_output_lngi_micro_iter_data_.append_micro_iter_data(
        {
            .current_xi_lambda_1_ = curr_interp_point.xi_lambda_1_,
            .current_xi_lambda_2_ = curr_interp_point.xi_lambda_2_,
            .current_xi_eigenvect_rot_ = curr_interp_point.xi_rel_eigenvect_rot_,
            .current_equiv_stress_ = state_quantities_.curr_equiv_stress_,
            .current_plastic_strain_ = plastic_strain_adapt_pred,
            .current_error_status_ = err_status,
        },
        lngi_step_counter - 1);
  }

  // write micro iteration data to csv
  if (parameter()->use_csv_output_lngi_micro_iter())
    csv_output_lngi_micro_iter_data_.write_lngi_micro_iter_data_to_csv();

  // DEBUG
  if (debug_mode(ele_gid_, gp_) && debug_pred_adapt)
  {
    std::cout << "End LNGI " << lnl_guess_interpolation_.num_of_lngi_ << " for ele_gid_ "
              << ele_gid_ << " and gp " << gp_ << std::endl;
    std::cout << "curr_interp_point: " << std::endl;
    lnl_guess_interpolation_.get_curr_interp_point(gp_).print(std::cout);
    std::cout << "lower_bound_interp_point: " << std::endl;
    lnl_guess_interpolation_.get_lower_bound_interp_point(gp_).print(std::cout);
    std::cout << "upper_bound_interp_point: " << std::endl;
    lnl_guess_interpolation_.get_upper_bound_interp_point(gp_).print(std::cout);
    std::cout << "equiv_stress:  " << state_quantities_.curr_equiv_stress_
              << ", plastic_strain: " << plastic_strain_adapt_pred
              << ", plastic_strain_rate: " << state_quantities_.curr_equiv_plastic_strain_rate_
              << std::endl;
  }


  // return adapted predictor
  return lnl_guess_interpolation_.guess_inv_plast_defgrad_;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::is_valid_local_newton_initial_guess(
    const Core::LinAlg::Matrix<3, 3>& defgrad, const Core::LinAlg::Matrix<3, 3>& inv_defgrad,
    const Core::LinAlg::Matrix<3, 3>& right_cg_tensor,
    const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_guess, const double plastic_strain_guess,
    ErrorType& err_status, StateQuantities& state_quantities,
    StateQuantityDerivatives& state_quantity_derivatives)
{
  // initialize error status
  err_status = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;

  // evaluate the current state with the adapted predictor
  state_quantities = evaluate_state_quantities(right_cg_tensor, inv_plastic_defgrad_guess,
      plastic_strain_guess, err_status, time_step_tracker_.dt_, StateQuantityEvalType::FullEval);

  // compute plastic strain rate: if 0.0 for
  // the current plasticity state, then we are "under" the yield
  // surface -> we stop the evaluation of the current state and
  // directly proceed to adapt the lower bound of the interpolation
  // factor
  if (std::abs(state_quantities.curr_equiv_plastic_strain_rate_ * time_step_tracker_.dt_) <=
      zero_plastic_strain_increment)
  {
    err_status = ErrorType::under_yield_surface;
  }

  // evaluate derivatives of the state quantities
  if (err_status == ErrorType::no_errors)
  {
    state_quantity_derivatives = evaluate_state_quantity_derivatives(right_cg_tensor,
        inv_plastic_defgrad_guess, plastic_strain_guess, err_status, time_step_tracker_.dt_,
        StateQuantityDerivEvalType::FullEval);
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradTransvIsotropElastViscoplast::get_line_search_step(
    const Core::LinAlg::Matrix<10, 1>& curr_sol, const Core::LinAlg::Matrix<3, 3>& CM,
    const Core::LinAlg::Matrix<10, 1>& curr_res, const Core::LinAlg::Matrix<10, 1>& incr,
    ErrorType& err_status)
{
  if (parameter()->use_csv_output_line_search_micro_iter())
  {
    // initialize micro iteration data for all "micro"
    // iterations of the subsequent line search, to be written to csv
    csv_output_line_search_micro_iter_data_ = CSVOutputLineSearchMicroIterData{
        CSVOutputTrackingData{.ele_gid_ = ele_gid_,
            .gp_ = gp_,
            .tn_ = (time_step_tracker_.tnp_ - time_step_tracker_.dt_),
            .tnp_ = time_step_tracker_.tnp_,
            .globiter_ = globiter_,
            .lnl_iter_ = lnl_data_.iter_ - 1}
        // we subtract 1 from the current local iteration number to start with 0, and
        // make this consistent with the output of the Local Newton Guess Interpolation
    };
  }

  // set necessary decrease parameter \f$ \rho \in \left(0, \frac{1}{2}\right) \f$ of the
  // backtracking algorithm
  const double rho = 1.0 / 4.0;

  // set line search decrease factor
  const double alpha_dec_fac = 0.9;
  // set maximum times we want to decrease the line search parameter,
  // before rejecting it
  const unsigned int max_dec_times = 200;

  // declare output line search parameter
  double alpha = 1.0;

  // declare updated solution (updated via the line search parameter)
  Core::LinAlg::Matrix<10, 1> next_sol{Core::LinAlg::Initialization::zero};
  next_sol.update(1.0, curr_sol, alpha, incr, 0.0);

  // set upper bound of the step size (maximum step size)
  double alpha_u = 1.0;

  // adapt the maximum step size to eventual negative plastic strains
  if (next_sol(9) < 0.0)
  {
    // check whether the current plastic strain is 0.0 and negative plastic strains are
    // obtained
    // -> this means, that any line search step size will fail. Hence, we directly return
    // with an error.
    if (curr_sol(9) == 0.0)
    {
      // csv runtime output
      if (parameter()->use_csv_output_line_search_micro_iter())
      {
        // append microiteration data
        csv_output_line_search_micro_iter_data_.append_micro_iter_data(
            CSVOutputLineSearchMicroIterData::MicroIterDataCollector{}, 0);

        // write microiteration data to csv
        csv_output_line_search_micro_iter_data_.write_line_search_micro_iter_data_to_csv();
      }

      err_status = ErrorType::failed_determ_line_search_step;
      return -1;
    }

    alpha_u = curr_sol(9) / std::abs(incr(9));
  }


  // save maximum value of the step size
  const double max_alpha = alpha_u;

  // set our current step size to the maximum step size
  alpha = alpha_u;
  // consistently update the next solution
  next_sol.update(1.0, curr_sol, alpha, incr, 0.0);

  // declare residual to be computed with the updated line search step
  Core::LinAlg::Matrix<10, 1> next_res{Core::LinAlg::Initialization::zero};

  // compute residual norm of the current iteration vector
  double curr_res_norm = curr_res.norm2();
  // square the residual of the current iteration vector in order to obtain the consistent
  // minimization function \f$ f_{\mathrm{curr}} = \| \boldsymbol{r}_{\mathrm{curr}} \|^2
  // \f$
  double curr_f = curr_res_norm * curr_res_norm;
  // compute residual norm of the next iteration vector
  double next_res_norm{1.0e8};
  // square the residual of the next iteration vector in order to obtain the consistent
  // minimization function \f$ f_{\mathrm{next}} = \| \boldsymbol{r}_{\mathrm{next}} \|^2
  // \f$
  double next_f{1.0e8};

  // compute squared increment (and 2-norm), used afterwards to check the
  // backtracking condition
  double incr_norm = incr.norm2();
  double incr_squared = incr_norm * incr_norm;

  // compute relative increment norm, i.e.,  \f$ \frac{\Delta
  // \boldsymbol{s}^{l}}{ \boldsymbol{s}^{l} } \f$
  double rel_incr_norm{incr_norm / next_sol.norm2()};

  // counter for the times we have decreased the line search parameter
  // in the backtracking algorithm
  unsigned int dec_times = 0;

  // initialize boolean for convergence check
  bool converged{false};

  // backtracking algorithm: decrease line search parameter until the
  // backtracking condition is met
  while (true)
  {
    // increment number of step size decrease procedures
    ++dec_times;

    // check whether we have decreased the line search parameter too many
    // times
    if (dec_times > max_dec_times)
    {
      // general local time integration analysis: set number of required iterations
      general_local_timint_analysis_utils.num_iters_and_steps_.eval_num_of_line_search_iters_ +=
          dec_times;

      // set error status (determination of line search step has failed)
      err_status = ErrorType::failed_determ_line_search_step;

      // csv runtime output
      if (parameter()->use_csv_output_line_search_micro_iter())
      {
        // set micro iteration data for the current evaluation
        csv_output_line_search_micro_iter_data_.append_micro_iter_data(
            {
                .current_alpha_ = alpha,
                .max_alpha_ = max_alpha,
                .current_equiv_stress_ = state_quantities_.curr_equiv_stress_,
                .current_plastic_strain_ = next_sol(9),
                .current_quadratic_residual_norm_ = -1,
                .max_quadratic_residual_norm_ = -1,
                .current_error_status_ = err_status,
            },
            dec_times - 1);
        // write micro iteration data to csv
        csv_output_line_search_micro_iter_data_.write_line_search_micro_iter_data_to_csv();
      }

      return -1.0;
    }

    // reset error status
    err_status = ErrorType::no_errors;

    // compute the residual associated with the considered step size
    // ("next iteration")
    next_res = calculate_local_newton_loop_residual(time_step_quantities_.current_rightCG_[gp_],
        next_sol, time_step_quantities_.last_plastic_defgrad_inverse_[gp_],
        time_step_quantities_.last_plastic_strain_[gp_], time_step_tracker_.dt_, err_status);

    if (err_status == ErrorType::no_errors)
    {
      // compute the corresponding residual norms and minimization functions
      next_res_norm = next_res.norm2();
      next_f = next_res_norm * next_res_norm;
    }
    else
    {
      // set micro iteration data for the current evaluation
      if (parameter()->use_csv_output_line_search_micro_iter())
      {
        csv_output_line_search_micro_iter_data_.append_micro_iter_data(
            {
                .current_alpha_ = alpha,
                .max_alpha_ = max_alpha,
                .current_equiv_stress_ = state_quantities_.curr_equiv_stress_,
                .current_plastic_strain_ = next_sol(9),
                .current_quadratic_residual_norm_ = -1,
                .max_quadratic_residual_norm_ = -1,
                .current_error_status_ = err_status,
            },
            dec_times - 1);
      }

      // decrease line search parameter
      alpha *= alpha_dec_fac;
      next_sol.update(1.0, curr_sol, alpha, incr, 0.0);

      continue;
    }

    // set micro iteration data for the current evaluation
    if (parameter()->use_csv_output_line_search_micro_iter())
    {
      csv_output_line_search_micro_iter_data_.append_micro_iter_data(
          {
              .current_alpha_ = alpha,
              .max_alpha_ = max_alpha,
              .current_equiv_stress_ = state_quantities_.curr_equiv_stress_,
              .current_plastic_strain_ = next_sol(9),
              .current_quadratic_residual_norm_ = next_f,
              .max_quadratic_residual_norm_ = curr_f - 2.0 * rho * alpha * incr_squared,
              .current_error_status_ = err_status,
          },
          dec_times - 1);
    }

    // update relative increment norm
    rel_incr_norm = alpha * incr_norm / next_sol.norm2();

    // first check for LNL convergence
    switch (lnl_data_.conv_check_)
    {
      case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::ResidualOnly:
        converged = (next_res_norm < lnl_data_.res_tol_);
        break;
      case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::IncrementOnly:
        converged = (rel_incr_norm < lnl_data_.incr_tol_);
        break;
      case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
          ResidualAndIncrement:
        converged = (next_res_norm < lnl_data_.res_tol_ && rel_incr_norm < lnl_data_.incr_tol_);
        break;
      default:
        FOUR_C_THROW("You should not be here (convergence checking of the Local Newton Loop)");
    }

    // check backtracking condition / LNL convergence
    if ((next_f < curr_f - 2.0 * rho * alpha * incr_squared) || converged)
    {
      // general local time integration analysis: set number of required iterations
      general_local_timint_analysis_utils.num_iters_and_steps_.eval_num_of_line_search_iters_ +=
          dec_times;

      // write micro iteration data to csv
      if (parameter()->use_csv_output_line_search_micro_iter())
        csv_output_line_search_micro_iter_data_.write_line_search_micro_iter_data_to_csv();

      err_status = ErrorType::no_errors;
      return alpha;
    }

    // decrease line search step size if the above backtracking checks were not successful
    alpha *= alpha_dec_fac;
    next_sol.update(1.0, curr_sol, alpha, incr, 0.0);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradTransvIsotropElastViscoplast::integrate_plastic_strain(
    const double equiv_stress, const double last_plastic_strain, const double dt,
    ErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::Matrix<2, 1> temp2x1{Core::LinAlg::Initialization::zero};

  // set predictor
  double plastic_strain = last_plastic_strain;

  // set loop settings
  unsigned int iter = 0;
  const unsigned int max_iter = 50;
  const double tol = 1.0e-8;
  double plastic_strain_rate = 1.0e10;
  double deriv_plastic_strain_rate = 1.0e10;
  double residual = 1.0e10;
  double jacobian = 1.0e10;

  // DEBUG
  if (debug_mode(ele_gid_, gp_) && debug_integrate_plastic_strain)
  {
    std::cout << std::string(50, '.') << std::endl;
    std::cout << "Integrating plastic strain starting from equiv_stress: " << equiv_stress
              << ", last_plastic_strain: " << last_plastic_strain << "\n";
  }


  // Newton-Raphson loop
  while (true)
  {
    // increment iterations
    ++iter;

    // check whether the maximum number of iterations was reached
    if (iter > max_iter)
    {
      std::cout << debug_get_error_info("WARNING: could not integrate plastic strain");
      err_status = ErrorType::overflow_error;
      return -1;
    }

    // DEBUG
    if (debug_mode(ele_gid_, gp_) && debug_integrate_plastic_strain)
    {
      std::cout << "iter: " << iter << " / " << max_iter << "\n";
      std::cout << "equiv_stress: " << equiv_stress << ", plastic_strain: " << plastic_strain
                << std::endl;
    }

    // compute plastic strain rate from the viscoplasticity law
    plastic_strain_rate = viscoplastic_law_->evaluate_plastic_strain_rate(
        equiv_stress, plastic_strain, dt, parameter()->max_plastic_strain_incr(), err_status);

    // return directly when encountering error
    if (err_status != ErrorType::no_errors)
    {
      // DEBUG
      if (debug_mode(ele_gid_, gp_) && debug_integrate_plastic_strain)
      {
        std::cout << "plastic strain rate could not be evaluated...-> we return with error status "
                  << EnumTools::enum_name(err_status) << std::endl;
      }



      return -1;
    }

    // compute residual
    residual = plastic_strain - last_plastic_strain - dt * plastic_strain_rate;


    // DEBUG
    if (debug_mode(ele_gid_, gp_) && debug_integrate_plastic_strain)
    {
      std::cout << "residual: " << residual << "\n";
    }



    // return solution
    if (std::abs(residual) < tol)
    {
      // DEBUG
      if (debug_mode(ele_gid_, gp_) && debug_integrate_plastic_strain)
      {
        std::cout << std::string(50, '.') << "\n";
      }


      return plastic_strain;
    }

    // compute derivative of the plastic strain rate w.r.t. plastic
    // strain
    temp2x1 = viscoplastic_law_->evaluate_derivatives_of_plastic_strain_rate(
        equiv_stress, plastic_strain, dt, parameter()->max_plastic_strain_deriv_incr(), err_status);
    deriv_plastic_strain_rate = temp2x1(1);

    // throw error
    if (err_status != ErrorType::no_errors)
    {
      // DEBUG
      if (debug_mode(ele_gid_, gp_) && debug_integrate_plastic_strain)
      {
        std::cout << "Plastic strain rate DERIVS could not be evaluated...->about to return"
                  << std::endl;
        std::cout << std::string(50, '.') << "\n";
      }

      return -1;
    }

    // compute jacobian
    jacobian = 1.0 - dt * deriv_plastic_strain_rate;

    // update solution
    plastic_strain -= residual / jacobian;


    // DEBUG
    if (debug_mode(ele_gid_, gp_) && debug_integrate_plastic_strain)
    {
      std::cout << "jacobian: " << jacobian << std::endl;
      std::cout << "plastic_strain: " << plastic_strain << std::endl;
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
ErrorAction Mat::InelasticDefgradTransvIsotropElastViscoplast::manage_evaluation_error(
    const ErrorType& err_status, Core::LinAlg::Matrix<10, 1>& sol,
    Core::LinAlg::Matrix<3, 3>& curr_CM)
{
  // return directly if there is no evaluation error
  if (err_status == ErrorType::no_errors)
  {
    return ErrorAction::continue_iteration;
  }


  // DEBUG
  if (debug_mode(ele_gid_, gp_) && debug_lnl)
  {
    std::cout << "manage_evaluation_error: " << EnumTools::enum_name(err_status) << "\n";
  }

  // general local time integration analysis: add error
  if (parameter()->analyze_timint() && general_local_timint_analysis_utils.increment_vars_)
    ++general_local_timint_analysis_utils.eval_error_map_[static_cast<ErrorType>(err_status)];

  // ERROR MANAGEMENT STRATEGY 1: substepping procedure
  if (parameter()->use_substepping())
  {
    const bool new_substep_status = prepare_new_substep(sol, curr_CM);
    if (!new_substep_status)
    {
      std::cout << debug_get_error_info(
                       "Could not find a suitable substep which converges for the given "
                       "settings!")
                << std::endl;
      return ErrorAction::return_solution_with_errors;  // return with error
    }
    return ErrorAction::next_iteration;
  }

  // ERROR MANAGEMENT STRATEGY 2: reset initial guess
  if (parameter()->use_lngi())
  {
    // increment number of Local Newton Guess Interpolations / Reinterpolations
    ++lnl_guess_interpolation_.num_of_lngi_;
    // general local time integration analysis: increment number of
    // reinterpolations
    if (parameter()->analyze_timint() && general_local_timint_analysis_utils.increment_vars_)
      ++general_local_timint_analysis_utils.num_iters_and_steps_.eval_num_of_lngi_reinterp_;


    // check whether Local Newton Guess Interpolation still possible
    if (!lnl_guess_interpolation_.is_interpolation_possible(gp_, 0))
    {
      std::cout << debug_get_error_info("Reinterpolation not possible") << std::endl;
      return ErrorAction::return_solution_with_errors;
    }


    // CHECK: Halve interval towards the elastic predictor, and check if this is
    // a possible initial guess
    // IF YES: accept as updated initial guess
    // IF NOT OR IF CURRENT INTERPOLATION FACTOR TOO NEAR TO LOWER BOUND \f$ \|
    // \overline{\sigma}(\xi) -
    // \overline{\sigma}(\xi_\text{l})
    // \| < \text{tol} \f$: set current interpolation factor as lower bound, and
    // reinterpolate initial guess with adapted bounds

    // get current interpolation point along with the bounds
    const LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
        lnl_guess_interpolation_.get_curr_interp_point(gp_);
    const LocalNewtonGuessInterpolation::InterpolationPoint interp_point_lbound =
        lnl_guess_interpolation_.get_lower_bound_interp_point(gp_);

    // build interpolation point to be evaluated next as initial guess
    LocalNewtonGuessInterpolation::InterpolationPoint next_interp_point =
        LocalNewtonGuessInterpolation::add_interpolation_points(
            0.5, interp_point_lbound, 0.5, curr_interp_point);

    // initialize boolean: is next interpolation point suitable as an initial
    // guess
    bool is_init_guess = true;


    // get deformation gradient and its inverse based on the saved values
    Core::LinAlg::Matrix<3, 3> defgrad{time_step_quantities_.current_defgrad_[gp_]};
    Core::LinAlg::Matrix<3, 3> inv_defgrad{Core::LinAlg::Initialization::zero};
    inv_defgrad.invert(defgrad);


    // compute inverse inelastic deformation gradients associated with the current lower
    // bound and with the interpolation point to be evaluated next
    Core::LinAlg::Matrix<3, 3> next_iFin = lnl_guess_interpolation_.interpolate_inv_plastic_defgrad(
        gp_, defgrad, next_interp_point, inv_defgrad);
    Core::LinAlg::Matrix<3, 3> lbound_iFin =
        lnl_guess_interpolation_.interpolate_inv_plastic_defgrad(
            gp_, defgrad, interp_point_lbound, inv_defgrad);


    // compute equivalent stresses associated with the lower bound and with the
    // interpolation point to be evaluated next interpolate inverse plastic deformation
    // gradient
    ErrorType next_equiv_stress_eval_err_status = ErrorType::no_errors;
    StateQuantities state_quantities_next =
        evaluate_state_quantities(curr_CM, next_iFin, 0.0, next_equiv_stress_eval_err_status,
            time_step_tracker_.dt_, StateQuantityEvalType::EquivStressOnly);
    ErrorType lbound_equiv_stress_eval_err_status = ErrorType::no_errors;
    StateQuantities state_quantities_lbound =
        evaluate_state_quantities(curr_CM, lbound_iFin, 0.0, lbound_equiv_stress_eval_err_status,
            time_step_tracker_.dt_, StateQuantityEvalType::EquivStressOnly);


    // decide whether this is a candidate for an initial guess based on the evaluability of
    // the equivalent stresses
    if (next_equiv_stress_eval_err_status == ErrorType::no_errors &&
        lbound_equiv_stress_eval_err_status == ErrorType::no_errors)
    {
      // both evaluable -> compute relative deviation between the stresses to decide whether
      // the next point should be evaluated next as a candidate for an initial guess
      // (relative deviation must be higher than a specified value)
      const double rel_deviation = std::abs(state_quantities_next.curr_equiv_stress_ -
                                            state_quantities_lbound.curr_equiv_stress_) /
                                   state_quantities_lbound.curr_equiv_stress_;

      is_init_guess = (rel_deviation > parameter()->lngi_reinterp_min_rel_dev());
    }
    else if (next_equiv_stress_eval_err_status == ErrorType::no_errors &&
             lbound_equiv_stress_eval_err_status != ErrorType::no_errors)
    {
      // next point is evaluable, but the lower bound is not -> candidate for an initial
      // guess
      is_init_guess = true;
    }
    else if (next_equiv_stress_eval_err_status != ErrorType::no_errors &&
             lbound_equiv_stress_eval_err_status != ErrorType::no_errors)
    {
      // neither the next point nor the lower bound are evaluable -> not a candidate for an
      // initial guess
      is_init_guess = false;
    }
    else
    {
      FOUR_C_THROW(
          "This does not make sense! The equivalent stress of the lower interpolation "
          "bound is "
          "evaluable but the point to be evaluated next is not!");
    }

    // --> now check whether the next point fully qualifies as an initial guess
    if (is_init_guess)
    {
      ErrorType next_full_eval_err_status =
          InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;

      // integrate plastic strain related to the next point
      const double next_plastic_strain = integrate_plastic_strain(
          state_quantities_next.curr_equiv_stress_, time_step_quantities_.last_plastic_strain_[gp_],
          time_step_tracker_.dt_, next_full_eval_err_status);

      if (next_full_eval_err_status ==
          InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
      {
        // evaluate as initial guess with integrated plastic strain
        is_valid_local_newton_initial_guess(defgrad, inv_defgrad, curr_CM, next_iFin,
            next_plastic_strain, next_full_eval_err_status, state_quantities_,
            state_quantity_derivatives_);

        // update initial guess if it is valid
        if (next_full_eval_err_status ==
            InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
        {
          lnl_guess_interpolation_.set_curr_interp_point(gp_, next_interp_point);

          sol = wrap_unknowns(next_iFin, next_plastic_strain);
        }
      }

      // if the next point is not suitable as an initial guess -> set dedicated boolean
      is_init_guess = (next_full_eval_err_status ==
                       InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors);
    }

    // if the next interpolation point is an initial guess: update sol (already
    // done, see above!);
    // otherwise: set as lower
    // bound, adapt interpolation interval and parameters --> and finally, reinterpolate
    // initial guess with updated interval bounds
    if (!is_init_guess)
    {
      // set lower bound
      lnl_guess_interpolation_.set_lower_bound_interp_point(gp_, next_interp_point);
      lnl_guess_interpolation_.set_curr_interp_point(gp_, next_interp_point);

      // adapt interpolation interval and interpolation parameter
      lnl_guess_interpolation_.adapt_interpolation_intervals(
          gp_, InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::overflow_error);
      lnl_guess_interpolation_.adapt_interpolation_parameters(gp_);

      // adapt initial guess: separate time tracking for Local Newton Guess Interpolation
      sol = interpolate_local_newton_guess(time_step_quantities_.current_defgrad_[gp_]);
    }

    // go to next iteration
    return ErrorAction::next_iteration;
  }

  // general local time integration analysis: write to csv
  if (parameter()->analyze_timint() && general_local_timint_analysis_utils.increment_vars_)
  {
    general_local_timint_analysis_utils.update_total();
    general_local_timint_analysis_utils.write_to_csv();
  }

  // return with errors if none of the error management strategies apply
  return ErrorAction::return_solution_with_errors;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::string Mat::InelasticDefgradTransvIsotropElastViscoplast::debug_get_error_info(
    const std::string& base_error_string)
{
  // auxiliaries
  std::ostringstream temp_ostream;

  // set output format for the numbers -> we can set it here for the
  // entire error message
  std::cout << std::fixed << std::setprecision(16) << std::endl;
  temp_ostream << std::fixed << std::setprecision(16) << std::endl;

  // declare the extended error message
  std::string extended_error_string{""};

  // get relevant error info
  extended_error_string += "BASE ERROR: \n";
  extended_error_string += base_error_string + "\n";
  extended_error_string +=
      "-> At EleID: " + std::to_string(ele_gid_) + ". At GP: " + std::to_string(gp_) + ".\n";
  extended_error_string += std::string(10, '.') + "\n";

  // add the material parameters
  extended_error_string += "PARAMETERS: \n";
  parameter()->raw_parameters().print(temp_ostream);
  viscoplastic_law_->parameter()->raw_parameters().print(temp_ostream);
  temp_ostream << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += std::string(10, '.') + "\n";

  // add the relevant last_ values
  extended_error_string += "LAST_ VALUES: \n";
  extended_error_string += "last_plastic_defgrad_inverse: \n";
  time_step_quantities_.last_plastic_defgrad_inverse_[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "last_substep_plastic_defgrad_inverse: \n";
  time_step_quantities_.last_substep_plastic_defgrad_inverse_[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "last_plastic_strain: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.last_plastic_strain_[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");

  extended_error_string += "last_plastic_strain_increment: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.last_plastic_strain_increment_[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");

  extended_error_string += "last_equiv_stress: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.last_equiv_stress_[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");

  extended_error_string += "last_equiv_stress_elastic_pred: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.last_equiv_stress_elastic_pred_[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");

  extended_error_string += "last_equiv_stress_plastic_pred: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.last_equiv_stress_plastic_pred_[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");

  extended_error_string += "last_substep_plastic_strain: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.last_substep_plastic_strain_[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "last_defgrad: \n";
  time_step_quantities_.last_defgrad_[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "last_rightCG: \n";
  time_step_quantities_.last_rightCG_[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += viscoplastic_law_->debug_get_error_info(gp_);
  extended_error_string += "last_xi_lambda_1 (Local Newton Guess Interpolation): \n";
  extended_error_string += "Double<1,1> \n";
  const LocalNewtonGuessInterpolation::InterpolationPoint last_interp_point =
      lnl_guess_interpolation_.get_last_interp_point(gp_);
  temp_ostream << last_interp_point.xi_lambda_1_ << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "last_xi_lambda_2 (Local Newton Guess Interpolation): \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << last_interp_point.xi_lambda_2_ << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "last_xi_eigenvect_rot (Local Newton Guess Interpolation): \n";
  extended_error_string += "array<3,1> \n";
  temp_ostream << last_interp_point.xi_rel_eigenvect_rot_[0] << std::endl;
  temp_ostream << last_interp_point.xi_rel_eigenvect_rot_[1] << std::endl;
  temp_ostream << last_interp_point.xi_rel_eigenvect_rot_[2] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "optimal_xi_lambda_1 (Local Newton Guess Interpolation): \n";
  extended_error_string += "Double<1,1> \n";
  const LocalNewtonGuessInterpolation::InterpolationPoint optimal_interp_point =
      lnl_guess_interpolation_.get_optimal_interp_point(gp_);
  temp_ostream << optimal_interp_point.xi_lambda_1_ << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "optimal_xi_lambda_2 (Local Newton Guess Interpolation): \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << optimal_interp_point.xi_lambda_2_ << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "optimal_xi_eigenvect_rot (Local Newton Guess Interpolation): \n";
  extended_error_string += "array<3,1> \n";
  temp_ostream << optimal_interp_point.xi_rel_eigenvect_rot_[0] << std::endl;
  temp_ostream << optimal_interp_point.xi_rel_eigenvect_rot_[1] << std::endl;
  temp_ostream << optimal_interp_point.xi_rel_eigenvect_rot_[2] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += std::string(10, '.') + "\n";

  // add the current values
  extended_error_string += "CURRENT_ VALUES: \n";
  extended_error_string += "current_defgrad: \n";
  time_step_quantities_.current_defgrad_[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "current_rightCG: \n";
  time_step_quantities_.current_rightCG_[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += std::string(10, '.');


  // add the current values
  extended_error_string += "OTHER VALUES: \n";
  extended_error_string += "dt: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_tracker_.dt_ << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += std::string(10, '.');


  return extended_error_string;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::debug_set_last_quantities(const int gp,
    const Core::LinAlg::Matrix<3, 3>& last_plastic_defgrad_inverse,
    const double last_plastic_strain, const double last_plastic_strain_increment,
    const double last_equiv_stress, const double last_equiv_stress_elastic_pred,
    const double last_equiv_stress_plastic_pred, const Core::LinAlg::Matrix<3, 3>& last_defgrad,
    const Core::LinAlg::Matrix<3, 3>& last_rightCG, const double last_xi_lambda_1,
    const double last_xi_lambda_2, const std::array<double, 3> last_xi_eigenvect_rot,
    const double last_max_xi_lambda_1, const double last_max_xi_lambda_2,
    const std::array<double, 3> last_max_xi_eigenvect_rot, const double optimal_xi_lambda_1,
    const double optimal_xi_lambda_2, const std::array<double, 3> optimal_xi_eigenvect_rot)
{
  time_step_quantities_.last_plastic_defgrad_inverse_[gp] = last_plastic_defgrad_inverse;
  time_step_quantities_.last_substep_plastic_defgrad_inverse_[gp] = last_plastic_defgrad_inverse;
  time_step_quantities_.last_plastic_strain_[gp] = last_plastic_strain;
  time_step_quantities_.last_plastic_strain_increment_[gp] = last_plastic_strain_increment;
  time_step_quantities_.last_substep_plastic_strain_[gp] = last_plastic_strain;
  time_step_quantities_.last_equiv_stress_[gp] = last_equiv_stress;
  time_step_quantities_.last_equiv_stress_elastic_pred_[gp] = last_equiv_stress_elastic_pred;
  time_step_quantities_.last_equiv_stress_plastic_pred_[gp] = last_equiv_stress_plastic_pred;
  time_step_quantities_.last_defgrad_[gp] = last_defgrad;
  time_step_quantities_.last_rightCG_[gp] = last_rightCG;
  lnl_guess_interpolation_.set_last_interp_point(
      gp, LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = last_xi_lambda_1,
              .xi_lambda_2_ = last_xi_lambda_2,
              .xi_rel_eigenvect_rot_ = last_xi_eigenvect_rot});
  lnl_guess_interpolation_.set_optimal_interp_point(
      gp, LocalNewtonGuessInterpolation::InterpolationPoint{.xi_lambda_1_ = optimal_xi_lambda_1,
              .xi_lambda_2_ = optimal_xi_lambda_2,
              .xi_rel_eigenvect_rot_ = optimal_xi_eigenvect_rot});

  // compute the material stretch and the rotation tensor for the
  // inverse inelastic defgrad
  Core::LinAlg::Matrix<3, 3> last_plastic_defgrad{Core::LinAlg::Initialization::zero};
  last_plastic_defgrad.invert(time_step_quantities_.last_plastic_defgrad_inverse_[gp]);
  Core::LinAlg::Matrix<3, 3> last_elastic_defgrad{Core::LinAlg::Initialization::zero};
  last_elastic_defgrad.multiply_nn(1.0, time_step_quantities_.last_defgrad_[gp],
      time_step_quantities_.last_plastic_defgrad_inverse_[gp], 0.0);

  // perform its polar-spectral decomposition
  Core::LinAlg::Matrix<3, 3> last_elastic_defgrad_rotation{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> last_elastic_defgrad_material_stretch{
      Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> last_elastic_stretch_eigenval{Core::LinAlg::Initialization::zero};
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> last_elastic_spectral_pairs;
  Core::LinAlg::matrix_3x3_polar_decomposition(last_elastic_defgrad, last_elastic_defgrad_rotation,
      last_elastic_defgrad_material_stretch, last_elastic_stretch_eigenval,
      last_elastic_spectral_pairs);

  // save "last" stretch components
  time_step_quantities_.last_elastic_stretch_eigenval_[gp] = {last_elastic_spectral_pairs[0].first,
      last_elastic_spectral_pairs[1].first, last_elastic_spectral_pairs[2].first};
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["lnl_iters"] = 1;
  names_and_size["inverse_plastic_defgrad"] = 9;
  names_and_size["plastic_strain"] = 1;
  names_and_size["plastic_strain_LNL"] =
      lnl_data_.max_iter_;  // plastic strain in each local iteration of the LNL
  names_and_size["equiv_stress"] = 1;
  names_and_size["equiv_stress_LNL"] = lnl_data_.max_iter_;
  names_and_size["iter_status_LNL"] = lnl_data_.max_iter_;
  names_and_size["residual_LNL"] = lnl_data_.max_iter_;
  names_and_size["defgrad"] = 9;
  names_and_size["rightCG"] = 9;
  names_and_size["xi_lambda_1"] = 1;
  names_and_size["xi_lambda_2"] = 1;
  names_and_size["xi_eigenvect_rot"] = 1;
  names_and_size["max_xi_lambda_1"] = 1;
  names_and_size["max_xi_lambda_2"] = 1;
  names_and_size["max_xi_eigenvect_rot"] = 1;
  viscoplastic_law_->register_output_data_names(names_and_size);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  // auxiliaries
  Core::LinAlg::Matrix<9, 1> temp9x1{Core::LinAlg::Initialization::zero};

  if (name == "lnl_iters")
  {
    for (int gp = 0; gp < static_cast<int>(lnl_data_.num_iter_curr_timestep_.size()); ++gp)
    {
      data(gp, 0) = lnl_data_.num_iter_curr_timestep_[gp];
    }
    return true;
  }
  else if (name == "inverse_plastic_defgrad")
  {
    for (int gp = 0;
        gp < static_cast<int>(time_step_quantities_.current_plastic_defgrad_inverse_.size()); ++gp)
    {
      Core::LinAlg::Voigt::matrix_3x3_to_9x1(
          time_step_quantities_.current_plastic_defgrad_inverse_[gp], temp9x1);

      for (int col = 0; col < 9; ++col)
      {
        data(gp, col) = temp9x1(col);
      }
    }
    return true;
  }
  else if (name == "plastic_strain")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_plastic_strain_.size());
        ++gp)
    {
      data(gp, 0) = time_step_quantities_.current_plastic_strain_[gp];
    }
    return true;
  }
  else if (name == "plastic_strain_LNL")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_plastic_strain_.size());
        ++gp)
    {
      for (unsigned int it = 0; it < lnl_data_.max_iter_; ++it)
      {
        data(gp, it) = lnl_data_.all_plastic_strain_[gp][it];
      }
    }
    return true;
  }
  else if (name == "equiv_stress")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_equiv_stress_.size());
        ++gp)
    {
      data(gp, 0) = time_step_quantities_.current_equiv_stress_[gp];
    }
    return true;
  }
  else if (name == "equiv_stress_LNL")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_plastic_strain_.size());
        ++gp)
    {
      for (unsigned int it = 0; it < lnl_data_.max_iter_; ++it)
      {
        data(gp, it) = lnl_data_.all_equiv_stress_[gp][it];
      }
    }
    return true;
  }
  else if (name == "iter_status_LNL")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_plastic_strain_.size());
        ++gp)
    {
      for (unsigned int it = 0; it < lnl_data_.max_iter_; ++it)
      {
        data(gp, it) = local_iteration_status_enum_to_double(lnl_data_.all_iter_status_[gp][it]);
      }
    }
    return true;
  }
  else if (name == "residual_LNL")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_plastic_strain_.size());
        ++gp)
    {
      for (unsigned int it = 0; it < lnl_data_.max_iter_; ++it)
      {
        data(gp, it) = lnl_data_.all_residual_[gp][it];
      }
    }
    return true;
  }

  else if (name == "defgrad")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_defgrad_.size()); ++gp)
    {
      Core::LinAlg::Voigt::matrix_3x3_to_9x1(time_step_quantities_.current_defgrad_[gp], temp9x1);

      for (int col = 0; col < 9; ++col)
      {
        data(gp, col) = temp9x1(col);
      }
    }
    return true;
  }
  else if (name == "rightCG")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_rightCG_.size()); ++gp)
    {
      Core::LinAlg::Voigt::matrix_3x3_to_9x1(time_step_quantities_.current_rightCG_[gp], temp9x1);


      for (int col = 0; col < 9; ++col)
      {
        data(gp, col) = temp9x1(col);
      }
    }
    return true;
  }
  else if (name == "xi_lambda_1")
  {
    for (int gp = 0;
        gp < static_cast<int>(time_step_quantities_.current_plastic_defgrad_inverse_.size()); ++gp)
    {
      LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
          lnl_guess_interpolation_.get_curr_interp_point(gp);
      data(gp, 0) = curr_interp_point.xi_lambda_1_;
    }
    return true;
  }
  else if (name == "xi_lambda_2")
  {
    for (int gp = 0;
        gp < static_cast<int>(time_step_quantities_.current_plastic_defgrad_inverse_.size()); ++gp)
    {
      LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
          lnl_guess_interpolation_.get_curr_interp_point(gp);
      data(gp, 0) = curr_interp_point.xi_lambda_2_;
    }
    return true;
  }
  else if (name == "xi_eigenvect_rot_comp_0")
  {
    for (int gp = 0;
        gp < static_cast<int>(time_step_quantities_.current_plastic_defgrad_inverse_.size()); ++gp)
    {
      LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
          lnl_guess_interpolation_.get_curr_interp_point(gp);
      data(gp, 0) = curr_interp_point.xi_rel_eigenvect_rot_[0];
    }
    return true;
  }
  else if (name == "xi_eigenvect_rot_comp_1")
  {
    for (int gp = 0;
        gp < static_cast<int>(time_step_quantities_.current_plastic_defgrad_inverse_.size()); ++gp)
    {
      LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
          lnl_guess_interpolation_.get_curr_interp_point(gp);
      data(gp, 0) = curr_interp_point.xi_rel_eigenvect_rot_[1];
    }
    return true;
  }
  else if (name == "xi_eigenvect_rot_comp_2")
  {
    for (int gp = 0;
        gp < static_cast<int>(time_step_quantities_.current_plastic_defgrad_inverse_.size()); ++gp)
    {
      LocalNewtonGuessInterpolation::InterpolationPoint curr_interp_point =
          lnl_guess_interpolation_.get_curr_interp_point(gp);
      data(gp, 0) = curr_interp_point.xi_rel_eigenvect_rot_[2];
    }
    return true;
  }


  // update information that we have Gauss point output for every global
  // iteration
  lnl_data_.is_Gauss_point_output_every_global_iter_ = true;

  return viscoplastic_law_->evaluate_output_data(name, data);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast::debug_set_linearization_type(
    const LinearizationType linearization_type)
{
  linearization_type_ = linearization_type;
}


FOUR_C_NAMESPACE_CLOSE
