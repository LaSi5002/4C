// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_mat_elast_couptransverselyisotropic.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <boost/graph/visitors.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


FOUR_C_NAMESPACE_OPEN


namespace Discret::Utils
{
  class FunctionOfTime;
}

namespace Mat
{
  using namespace InelasticDefgradTransvIsotropElastViscoplastUtils;

  namespace PAR
  {
    enum class InelasticSource;

    /*----------------------------------------------------------------------*/
    /*! \class InelasticDeformationDirection
     *
     * Calculates and holds growth direction in matrix format for anisotropic growth
     */
    class InelasticDeformationDirection
    {
     public:
      /*!
       * @brief standard constructor
       * @param[in] growthdirection  direction of anisotropic growth
       */
      explicit InelasticDeformationDirection(std::vector<double> growthdirection);

      /// reference to matrix that determines growth direction
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& growth_dir_mat() const
      {
        return growth_dir_mat_;
      }

     private:
      /// matrix that determines growth direction
      Core::LinAlg::SymmetricTensor<double, 3, 3> growth_dir_mat_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradNoGrowth
     *
     * This is a parameter class that is only needed to implement the pure virtual method
     * 'create_material()'.
     */
    class InelasticDefgradNoGrowth : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradNoGrowth(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; }
    };


    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradScalar
     *
     * This is a parameter class holding parameters for evaluation of inelastic deformation (incl.
     * linearization) induced by a scalar.
     */
    class InelasticDefgradScalar : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradScalar(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; }

      /// scalar that causes growth
      int scalar1() const { return scalar1_; }

      //! concentration, at which no growth occurs
      double scalar1_ref_conc() const { return scalar1_ref_conc_; }

     private:
      /// scalar that causes growth
      const int scalar1_;

      //! concentration, at which no growth occurs
      const double scalar1_ref_conc_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradTimeFunct
     *
     * This is a parameter class holding parameters for evaluation of inelastic deformation induced
     * by a given time-dependent function.
     */
    class InelasticDefgradTimeFunct : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradTimeFunct(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; }

      /// function number that sets determinant of inelastic def. grad.
      int funct_num() const { return funct_num_; }

     private:
      /// function number that sets determinant of inelastic def. grad.
      const int funct_num_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradLinScalar
     *
     * This is a specialized parameter class that holds the growth factor for linear growth
     */
    class InelasticDefgradLinScalar : public InelasticDefgradScalar
    {
     public:
      /// standard constructor
      explicit InelasticDefgradLinScalar(const Core::Mat::PAR::Parameter::Data& matdata);

      //! molar factor that causes growth
      double scalar1_molar_growth_fac() { return scalar1_molar_growth_fac_; }

     private:
      //! molar factor that causes growth
      const double scalar1_molar_growth_fac_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradLinScalarAniso
     *
     * This is a specialized parameter class that can return the anisotropic growth direction
     * represented as a growth matrix
     */
    class InelasticDefgradLinScalarAniso : public InelasticDefgradLinScalar
    {
     public:
      /// standard constructor
      explicit InelasticDefgradLinScalarAniso(const Core::Mat::PAR::Parameter::Data& matdata);

      /// reference to matrix that determines growth direction
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& growth_dir_mat()
      {
        return growth_dir_->growth_dir_mat();
      }

     private:
      /// calculation of direction of inelastic deformation
      std::shared_ptr<const InelasticDeformationDirection> growth_dir_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradIntercalFrac
     *
     * This parameter class provides all electrochemical quantities that are needed to calculate the
     * intercalation fraction from a given species concentration.
     */
    class InelasticDefgradIntercalFrac : public InelasticDefgradScalar
    {
     public:
      explicit InelasticDefgradIntercalFrac(const Core::Mat::PAR::Parameter::Data& matdata);

      /// saturation concentration of material
      double cmax() const { return c_max_; }
      /// intercalation fraction at saturation concentration of material
      double chimax() const { return chi_max_; }

     private:
      /// saturation concentration of material
      double c_max_;
      /// intercalation fraction at saturation concentration of material
      double chi_max_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradPolyIntercalFrac
     *
     * This parameter class provides the value of the polynomial that models the growth evaluated in
     * the reference configuration.
     */
    class InelasticDefgradPolyIntercalFrac : public InelasticDefgradIntercalFrac
    {
     public:
      explicit InelasticDefgradPolyIntercalFrac(const Core::Mat::PAR::Parameter::Data& matdata);

      /// return value of polynomial at reference intercalation fraction
      double get_polynom_reference_value() const { return polynom_reference_value_; }

      // set value of polynomial at reference intercalation fraction
      void set_polynom_reference_value(double polynomReferenceValue)
      {
        polynom_reference_value_ = polynomReferenceValue;
      }

      //! polynomial coefficients that describe the growth law
      std::vector<double> poly_coeffs() const { return poly_coeffs_; }

      //! upper bound of polynomial
      double x_max() const { return x_max_; }

      //! lower bound of polynomial
      double x_min() const { return x_min_; }

     private:
      const std::vector<double> poly_coeffs_;

      /// value of polynomial at reference intercalation fraction
      double polynom_reference_value_;

      //! upper bound of polynomial
      const double x_max_;

      //! lower bound of polynomial
      const double x_min_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradPolyIntercalFracAniso
     *
     * This is a specialized parameter class that can return the anisotropic growth direction
     * represented as a growth matrix
     */
    class InelasticDefgradPolyIntercalFracAniso : public InelasticDefgradPolyIntercalFrac
    {
     public:
      /// standard constructor
      explicit InelasticDefgradPolyIntercalFracAniso(
          const Core::Mat::PAR::Parameter::Data& matdata);

      /// return reference to matrix that determines growth direction
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& growth_dir_mat() const
      {
        return growth_dir_->growth_dir_mat();
      };

     private:
      /// pointer to object, that calculates and holds direction of inelastic deformation
      std::shared_ptr<InelasticDeformationDirection> growth_dir_;
    };

    /*----------------------------------------------------------------------
    ----------------------------------------------------------------------*/
    /*! \class InelasticDefgradLinTempIso

    Parameter class of InelasticDefgradLinTempIso.
    */
    class InelasticDefgradLinTempIso : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradLinTempIso(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      /// return temperature related growth factor
      double get_temp_growth_fac() const { return temp_growth_fac_; };

      /// return value of temperature that causes no growth
      double ref_temp() const { return ref_temp_; };

     private:
      /// value of temperature that causes no growth
      const double ref_temp_;

      /// growth factor
      const double temp_growth_fac_;
    };

    /*----------------------------------------------------------------------
    ---------------------------------------------------------------------*/
    /*! \class InelasticDefgradTransvIsotropElastViscoplast
     * Parameter class of InelasticDefgradTransvIsotropElastViscoplast.
     */
    class InelasticDefgradTransvIsotropElastViscoplast : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradTransvIsotropElastViscoplast(
          const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      //! get ID of the viscoplasticity law
      [[nodiscard]] int viscoplastic_law_id() const { return viscoplastic_law_id_; };
      //! get global ID of the fiber reader material
      [[nodiscard]] int fiber_reader_gid() const { return fiber_reader_gid_; };
      //! get yield condition parameter \f[ A \f]
      [[nodiscard]] double yield_cond_a() const { return yield_cond_a_; };
      //! get yield condition parameter \f[ B \f]
      [[nodiscard]] double yield_cond_b() const { return yield_cond_b_; };
      //! get yield condition parameter \f[ F \f]
      [[nodiscard]] double yield_cond_f() const { return yield_cond_f_; };
      //! get material behavior
      [[nodiscard]] MatBehavior mat_behavior() const { return mat_behavior_; };
      //! get boolean: should Local Newton Guess Interpolation be used?
      [[nodiscard]] bool use_lngi() const { return use_lngi_; };
      //! get type of eigenvalue assignment for the elastic stretch within the plastic predictor
      [[nodiscard]] LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvalType
      lngi_plastic_pred_elastic_stretch_eigenval_type() const
      {
        return lngi_plastic_pred_elastic_stretch_eigenval_type_;
      };
      //! get type of eigenvector rotation assignment for the elastic stretch within the plastic
      //! predictor
      [[nodiscard]] LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvectRotType
      lngi_plastic_pred_elastic_stretch_eigenvect_rot_type() const
      {
        return lngi_plastic_pred_elastic_stretch_eigenvect_rot_type_;
      };
      //! get type of rotation assignment within the plastic
      //! predictor
      [[nodiscard]] LocalNewtonGuessInterpolation::PlasticPredictorRotationType
      lngi_plastic_pred_rot_type() const
      {
        return lngi_plastic_pred_rot_type_;
      };
      //! get boolean: check consistency of the matrices and their
      //! components determined and analyzed during Local Newton Guess Interpolation? (true: yes,
      //! false: no)
      [[nodiscard]] bool lngi_check_consistency() const { return lngi_check_consistency_; };
      //! get boolean: precondition matrices for the Local Newton Guess Interpolation, i.e., set
      //! components smaller than a set numerical tolerance to 0? (true: yes, false: no)
      [[nodiscard]] bool lngi_precondition_matrices() const { return lngi_precondition_matrices_; };
      //! get numerical tolerance used to precondition matrices for the Local Newton Guess
      //! Interpolation, i.e., set components smaller than (numerical tolerance * 2-norm of input
      //! matrix) to 0?
      [[nodiscard]] double lngi_precondition_matrices_num_tol() const
      {
        return lngi_precondition_matrices_num_tol_;
      };
      //! get starting point type for the Local Newton Guess Interpolation
      [[nodiscard]] LocalNewtonGuessInterpolation::LocalNewtonGuessInterpolationStartingPointType
      lngi_starting_point_type() const
      {
        return lngi_starting_point_type_;
      };
      //! get user-specified starting point for Local Newton Guess Interpolation, when using the
      //! user_set starting point type
      [[nodiscard]] double lngi_starting_point() const { return lngi_starting_point_; }
      //! get user-specified interval scanning parameter for the Local
      // Newton Guess Interpolation
      [[nodiscard]] double lngi_interval_scan_param() const { return lngi_interval_scan_param_; }
      //! get maximum number of Local Newton Guess Reinterpolations to be used within a
      //! single Local Newton Loop
      [[nodiscard]] int lngi_max_num_reinterp() const { return lngi_max_num_reinterp_; }
      //! get minimum interpolation interval | xi_upper - xi_lower | (2-norm in
      //! interpolation space) for the Local Newton Guess Interpolation, for which further
      //! interpolation is not possible / feasible
      [[nodiscard]] double lngi_min_interp_interval() const { return lngi_min_interp_interval_; }
      //! get minimum relative deviation between equivalent stresses related to the current
      //! interpolation point and its lower bound  | \overline{\xi} -
      //! \overline{\sigma}(\xi_lower) | / \overline{\sigma}(\xi_lower) for the Local Newton Guess
      //! Reinterpolation, upon which xi_lower is set as xi in the reinterpolation routine
      [[nodiscard]] double lngi_reinterp_min_rel_dev() const { return lngi_reinterp_min_rel_dev_; }
      //! get boolean: output relevant data from each microiteration of the Local Newton Guess
      //! Interpolation (and Reinterpolations) to a dedicated csv file?
      [[nodiscard]] bool use_csv_output_lngi_micro_iter()
      {
        return use_csv_output_lngi_micro_iter_;
      }

      //! get boolean:       use steepest descent direction if the Newton
      //! direction fails in single Local Newton iterations? (true: yes, false: no)
      [[nodiscard]] bool use_steepest_descent_update_correction() const
      {
        return use_steepest_descent_update_correction_;
      }
      //! get boolean: use line search to avoid negative plastic strains
      //! in the Local Newton Loop? (true: yes, false: no)
      [[nodiscard]] bool use_line_search() const { return use_line_search_; };
      //! get boolean: check angle condition prior to backtracking line
      //! search? (true: yes, false: no)
      [[nodiscard]] bool check_line_search_angle_condition() const
      {
        return check_line_search_angle_condition_;
      };
      //! tolerance for angle condition prior to backtracking line
      //! search? (true: yes, false: no)
      [[nodiscard]] double line_search_angle_condition_tolerance() const
      {
        return line_search_angle_condition_tolerance_;
      };
      //! get boolean: use substepping in the time integration scheme? (true: yes, false: no)
      [[nodiscard]] bool use_substepping() const { return use_substepping_; };
      //! get boolean: analyze time integration scheme and write
      //! output to csv? (true: yes, false: no)
      [[nodiscard]] bool analyze_timint() const { return analyze_timint_; };
      //! get relative timer tolerance (analyze_timint_ = True) for the return mapping evaluation in
      //! the current timestep
      [[nodiscard]] double analyze_timint_timer_rel_tol() const
      {
        return analyze_timint_timer_rel_tol_;
      };
      //! get maximum number of times a time step can be halved into smaller and smaller substeps
      [[nodiscard]] unsigned int max_halve_number() const
      {
        return static_cast<unsigned int>(max_substepping_halve_num_);
      }
      //! get the type of time integration for the evolution equations
      //! of history variables
      [[nodiscard]] TimIntType timint_type() const { return timint_type_; };
      //! get the type of material linearization used
      [[nodiscard]] LinearizationType linearization_type() const { return linearization_type_; };
      //! DEBUG: set linearization type
      void debug_set_linearization_type(const LinearizationType linearization_type);
      //! get maximum, numerically evaluable plastic strain increment
      [[nodiscard]] double max_plastic_strain_incr() const { return max_plastic_strain_incr_; };
      //! get maximum, numerically evaluable value for the increment of
      //! the plastic strain derivatives (dt * derivative)
      [[nodiscard]] double max_plastic_strain_deriv_incr() const
      {
        return max_plastic_strain_deriv_incr_;
      }
      //! get computation method for the matrix exponential
      [[nodiscard]] Core::LinAlg::MatrixExpCalcMethod mat_exp_calc_method() const
      {
        return mat_exp_calc_method_;
      }
      //! get computation method for the first derivative of the matrix exponential
      [[nodiscard]] Core::LinAlg::GenMatrixExpFirstDerivCalcMethod mat_exp_deriv_calc_method() const
      {
        return mat_exp_deriv_calc_method_;
      }
      //! get computation method for the matrix logarithm
      [[nodiscard]] Core::LinAlg::MatrixLogCalcMethod mat_log_calc_method() const
      {
        return mat_log_calc_method_;
      }
      //! get computation method for the first derivative of the matrix logarithm
      [[nodiscard]] Core::LinAlg::GenMatrixLogFirstDerivCalcMethod mat_log_deriv_calc_method() const
      {
        return mat_log_deriv_calc_method_;
      }
      //! get boolean: output relevant data from each iteration of the last, failed Local Newton
      //! loop to a dedicated csv file?
      [[nodiscard]] bool use_csv_output_failed_local_newton_iter() const
      {
        return use_csv_output_failed_local_newton_iter_;
      }

      //! get boolean: output relevant data from each microiteration of the line
      //! search algorithm(s) to a dedicated csv file?
      [[nodiscard]] bool use_csv_output_line_search_micro_iter()
      {
        return use_csv_output_line_search_micro_iter_;
      }

      //! get convergence tolerance for the Local Newton-Raphson scheme
      //! (absolute residual value)
      [[nodiscard]] double local_newton_res_tol() { return local_newton_res_tol_; }

      //! get convergence tolerance for the Local Newton-Raphson scheme
      //! (2-norm of solution increment)
      [[nodiscard]] double local_newton_incr_tol() { return local_newton_incr_tol_; }

      //! get convergence check strategy for the Local Newton-Raphson scheme
      [[nodiscard]] LocalNewtonConvCheck local_newton_conv_check()
      {
        return local_newton_conv_check_;
      }

      //! strategy in case of divergence of the Local Newton-Raphson scheme
      [[nodiscard]] LocalNewtonDiverCont local_newton_diver_cont()
      {
        return local_newton_diver_cont_;
      }

     private:
      //! ID of the viscoplasticity law
      const int viscoplastic_law_id_;

      //! global ID of the material used for fiber reading (transversely isotropic)
      const int fiber_reader_gid_;

      //! yield condition parameter \f[ A \f]
      const double yield_cond_a_;
      //! yield condition parameter \f[ B \f]
      const double yield_cond_b_;
      //! yield condition parameter \f[ F \f]
      const double yield_cond_f_;

      //! material behavior (transversely isotropic or isotropic)
      const MatBehavior mat_behavior_;

      //! computation method for the time integration of the
      //! history variables
      const TimIntType timint_type_;

      //! linearization method
      LinearizationType linearization_type_;

      //! maximum, numerically evaluable plastic strain increment
      const double max_plastic_strain_incr_;

      //! maximum, numerically evaluable increment of
      //! plastic strain derivatives (time_step * derivative)
      const double max_plastic_strain_deriv_incr_;

      //! boolean: use Local Newton Guess Interpolation?
      const bool use_lngi_;

      //! Local Newton Guess Interpolation: type of elastic stretch eigenvalue assignment within
      //! plastic predictor
      const LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvalType
          lngi_plastic_pred_elastic_stretch_eigenval_type_;

      //! Local Newton Guess Interpolation: type of eigenvector rotation assignment for the elastic
      //! stretch within the plastic predictor
      const LocalNewtonGuessInterpolation::PlasticPredictorElasticStretchEigenvectRotType
          lngi_plastic_pred_elastic_stretch_eigenvect_rot_type_;

      //! Local Newton Guess Interpolation: type of eigenvector rotation assignment for the elastic
      //! stretch within the plastic predictor
      const LocalNewtonGuessInterpolation::PlasticPredictorRotationType lngi_plastic_pred_rot_type_;

      //! Local Newton Guess Interpolation: starting point type
      //! for the Local Newton Guess Interpolation
      const LocalNewtonGuessInterpolation::LocalNewtonGuessInterpolationStartingPointType
          lngi_starting_point_type_;

      //! Local Newton Guess Interpolation: value for the starting
      // point of the Local Newton Guess Interpolation to be used for the
      // user_set starting point type.
      const double lngi_starting_point_;

      //! Local Newton Guess Interpolation: interval scanning parameter $k_\mathrm{scan}$
      const double lngi_interval_scan_param_;

      //! Local Newton Guess Interpolation: maximum number of Local Newton Guess Reinterpolations
      //! allowed in a single Local Newton Loop until error is thrown
      const int lngi_max_num_reinterp_;

      //! Local Newton Guess Interpolation: minimum interpolation interval | xi_upper - xi_lower |
      //! (2-norm in interpolation space), for which further interpolation is not possible /
      //! feasible
      const double lngi_min_interp_interval_;

      //! Local Newton Guess Reinterpolation: get minimum relative deviation between equivalent
      //! stresses related to the current interpolation point and its lower bound  | \overline{\xi}
      //! -
      //! \overline{\sigma}(\xi_lower) | / \overline{\sigma}(\xi_lower) upon which xi_lower is set
      //! as xi in the reinterpolation routine
      const double lngi_reinterp_min_rel_dev_;

      //! Local Newton Guess Interpolation: precondition matrices,
      //! i.e., set components smaller than a set numerical
      //! tolerance to 0? (true: yes, false: no)
      const bool lngi_precondition_matrices_;

      //! Local Newton Guess Interpolation: numerical tolerance used to precondition matrices, i.e.,
      //! set components smaller than (numerical tolerance
      //! * 2-norm of input matrix) to 0? (true: yes, false: no)
      const double lngi_precondition_matrices_num_tol_;

      //! Local Newton Guess Interpolation: check consistency of the matrices and their
      //! components determined and analyzed during the interpolation algorithm? (true: yes, false:
      //! no)
      const bool lngi_check_consistency_;

      //! boolean: use steepest descent direction if the Newton
      //! direction fails in single Local Newton iterations? (true: yes, false: no)
      const double use_steepest_descent_update_correction_;

      //! boolean: use line search to avoid negative plastic strains in
      //! the Local Newton Loop? (true: yes, false: no)
      const bool use_line_search_;

      //! boolean: check angle condition prior to backtracking line
      //! search algorithm (see Andrei: Modern Numerical Nonlinear
      //! Optimization, Springer, p. 46-48)
      const bool check_line_search_angle_condition_;

      //! tolerance for the angle condition prior to backtracking line
      //! search algorithm (see Andrei: Modern Numerical Nonlinear
      //! Optimization, Springer, p. 46-48)
      const double line_search_angle_condition_tolerance_;

      //! boolean: use substepping? (true: yes, false: no)
      const bool use_substepping_;

      //! boolean: analyze time integration and write output to csv?
      const bool analyze_timint_;

      //! relative timer tolerance (analyze_timint_ = True) for the return mapping evaluation in the
      //! current timestep -> return mapping / determination of inelastic defgrad is repeated until
      //! the computation time changes only within the set relative tolerance
      const double analyze_timint_timer_rel_tol_;

      //! maximum number of times the given time step can be halved before reaching the minimum
      //! allowed substep length
      const int max_substepping_halve_num_;

      //! utilized computation method for the matrix exponential
      const Core::LinAlg::MatrixExpCalcMethod mat_exp_calc_method_;

      //! utilized computation method for the first derivative of the matrix exponential
      const Core::LinAlg::GenMatrixExpFirstDerivCalcMethod mat_exp_deriv_calc_method_;

      //! utilized computation method for the matrix logarithm
      const Core::LinAlg::MatrixLogCalcMethod mat_log_calc_method_;

      //! utilized computation method for the first derivative of the matrix logarithm
      const Core::LinAlg::GenMatrixLogFirstDerivCalcMethod mat_log_deriv_calc_method_;

      //! convergence tolerance for the Local Newton-Raphson scheme
      //! (absolute residual value)
      const double local_newton_res_tol_;

      //! convergence tolerance for the Local Newton-Raphson scheme
      //! (2-norm of solution increment)
      const double local_newton_incr_tol_;

      //! convergence check strategy for the Local Newton-Raphson scheme
      const LocalNewtonConvCheck local_newton_conv_check_;

      //! strategy in case of divergence of the Local Newton-Raphson scheme
      const LocalNewtonDiverCont local_newton_diver_cont_;

      //! output relevant data from each iteration of the last, failed Local Newton
      //! loop to a dedicated csv file
      const bool use_csv_output_failed_local_newton_iter_;

      //! output relevant data from each microiteration of the Local Newton Guess Interpolation (and
      //! Reinterpolations) to a dedicated csv file
      const bool use_csv_output_lngi_micro_iter_;

      //! output relevant data from each microiteration of the line
      //! search algorithm(s) to a dedicated csv file
      const bool use_csv_output_line_search_micro_iter_;
    };
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradLinearShape
   *
   * This class provides the functionality to be used if the growth law obeys a linear relation
   */
  class InelasticDefgradLinearShape
  {
   public:
    /*!
     * @brief constructor with required parameters
     *
     * @param[in] growth_fac       linear growth factor (slope of linear function)
     * @param[in] reference_value  reference value
     */
    explicit InelasticDefgradLinearShape(double growth_fac, double reference_value);

    /*!
     * @brief evaluation of the linear growth law
     *
     * @param[in] value           value the linear relation shall be evaluated for
     * @return growth factor
     */
    double evaluate_linear_growth(double value) const;

    /// growth factor (needed for linearizations)
    double growth_fac() const { return growth_fac_; }

   private:
    /// growth factor
    const double growth_fac_;

    /// reference value
    const double reference_value_;
  };  // namespace Mat

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradPolynomialShape
   *
   * This class provides the functionality to be used if the growth law obeys a polynomial relation
   */
  class InelasticDefgradPolynomialShape
  {
   public:
    /*!
     * @brief  constructor with required parameters
     *
     * @param[in] poly_coeffs  coefficients describing the polynomial to be evaluated
     * @param[in] x_min        lower bound of validity of the polynomial
     * @param[in] x_max        upper bound of validity of the polynomial
     */
    explicit InelasticDefgradPolynomialShape(
        std::vector<double> poly_coeffs, double x_min, double x_max);

    /*!
     * @brief checks the bounds of validity of the polynomial and writes a warning to screen if
     * bounds are violated
     *
     * @param[in] x  value the polynomial is evaluated at
     */
    void check_polynomial_bounds(double x) const;

    /*!
     * @brief Evaluate the polynomial defined by #PolyCoeffs_ at the current position X
     *
     * @param[in] x  value the polynomial is evaluated at
     * @return value of the polynomial evaluated at x
     */
    double compute_polynomial(double x);

    /*!
     * @brief Evaluate the first derivative of the polynomial defined by #PolyCoeffs_ at the current
     * position x
     *
     * @param[in] x  value the polynomial is evaluated at
     * @return value the first derivative of the polynomial evaluated at x
     */
    double compute_polynomial_derivative(double x);

   private:
    /// coefficients of the polynomial to be evaluated
    const std::vector<double> poly_coeffs_;
    /// lower bound of validity of polynomial
    const double x_min_;
    /// upper bound of validity of polynomial
    const double x_max_;
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradFactors

      Provides the interface called by the class "MultiplicativeSplitDefgrad_ElastHyper"
      and is needed to evaluate the inelastic deformation gradient and
      their derivatives w.r.t. the primary variables.

      In the material "MultiplicativeSplitDefgrad_ElastHyper" the deformation gradient is split
      multiplicatively in elastic and inelastic deformation gradients (F = F_{el} * F_{in}).
      The inelastic deformation gradient itself can be a product of different inelastic
      deformation gradients, i.e. F_{in} = F_{in,1} * F_{in,2} * ... * F_{in,n}.
      The derived classes below are needed to evaluate the inverse of the j-th inelastic
      deformation gradient F_{in,j}^{-1} and its derivatives w.r.t. the primary variables.
  */
  class InelasticDefgradFactors
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~InelasticDefgradFactors() = default;

    /// construct material with specific material params
    explicit InelasticDefgradFactors(Core::Mat::PAR::Parameter* params);

    /*!
     * @brief create object by input parameter ID
     *
     * @param[in] matnum  material ID
     * @return pointer to material that is defined by material ID
     */
    static std::shared_ptr<InelasticDefgradFactors> factory(int matnum);

    /// provide material type
    virtual Core::Materials::MaterialType material_type() const = 0;

    /*!
     * @brief evaluate the inelastic deformation gradient and its inverse
     *
     * @param[in] defgrad  Deformation gradient
     * @param[in] iFin_other Already computed inverse inelastic deformation gradient
     *              (from already computed inelastic factors in the multiplicative split material)
     * @param[out] iFinM   Inverse inelastic deformation gradient
     */
    virtual void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) = 0;

    /*!
     * @brief evaluate additional terms for the elasticity tensor
     *
     * @param[in] defgrad  Deformation gradient
     * @param[in] iFin_other Already computed inverse inelastic deformation gradient
     *              (from already computed inelastic factors in the multiplicative split material)
     * @param[in] iFinjM   Inverse inelastic deformation gradient of current inelastic contribution
     *                     as 3x3 matrix
     * @param[in] iCV      Inverse right Cauchy-Green tensor
     * @param[in] dSdiFinj Derivative of 2nd Piola Kirchhoff stresses w.r.t. the inverse inelastic
     *                     deformation gradient of current inelastic contribution
     * @param[in,out] cmatadd  Additional elasticity tensor
     */
    virtual void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) = 0;

    /*!
     * @brief calculate the derivative of the inelastic deformation gradient
     *
     * @param[in] detjacobian  determinant of the deformation gradient
     * @param[out] dFindx      derivative of inelastic deformation gradient w.r.t. primary variable
     *                         of different field
     */
    virtual void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) = 0;

    /*!
     * @brief evaluate off-diagonal stiffness matrix for monolithic systems to get the
     *        cross-linearizations
     *
     * @param[in] defgrad Deformation gradient
     * @param[in] iFinjM  Inverse inelastic deformation gradient of current inelastic contribution
     *                    as 3x3 matrix
     * @param[in] dSdiFinj  Derivative of 2nd Piola Kirchhoff stresses w.r.t. the inverse inelastic
     *                      deformation gradient of current inelastic contribution
     * @param[in,out] dstressdx Derivative of 2nd Piola Kirchhoff stresses w.r.t. primary variable
     *                          of different field
     */
    virtual void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) = 0;

    /*!
     * @brief pre-evaluation, intended to be used for stuff that has to be done only once per
     *        evaluate()
     *
     * @param[in] params  parameter list as handed in from the element
     * @param[in] gp      Gauss point
     * @param[in] eleGID  Element ID
     */
    virtual void pre_evaluate(const Teuchos::ParameterList& params,
        const EvaluationContext& context, int gp, int eleGID) = 0;

    /*!
     * @brief set gauss point concentration to parameter class
     *
     * @param[in] concentration  gauss point concentration to be set to internal member of parameter
     *                           class
     *
     * @note This method is used by methods called from the contact algorithm. Since the gauss point
     * ids do not match anyways (volume vs. surface element gauss point ids) and the id is not
     * relevant since the method is only called for one gauss point anyways, we set it to a dummy
     * gauss point id of 0 here
     */
    virtual void set_concentration_gp(double concentration) {};

    /// return material parameters
    virtual Core::Mat::PAR::Parameter* parameter() const { return params_; }

    /// Get type of scalar, that leads to deformation
    virtual PAR::InelasticSource get_inelastic_source() = 0;

    /*!
     * @brief Setup inelastic defgrad factor for the specific element
     *
     * @param[in] numgp Number of Gauss points
     * @param[in] container Input parameter Container
     */
    virtual void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) = 0;

    /// update history variables of the inelastic defgrad factors for next time step
    virtual void update() = 0;

    virtual void pack_inelastic(Core::Communication::PackBuffer& data) const = 0;

    virtual void unpack_inelastic(Core::Communication::UnpackBuffer& data) = 0;

    /*!
     * @brief Register names of the internal data that should be saved during runtime output
     *
     * @param[out] name_and_size Unordered map of names of the data with the respective vector size
     */
    virtual void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const
    {
    }

    /*!
     * @brief Evaluate internal data for every Gauss point saved for output during runtime
     * output
     *
     * @param[in] name  Name of the data to export
     * @param[out] data NUMGPxNUMDATA Matrix holding the data
     *
     * @return true if data is set by the material, otherwise false
     */
    virtual bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
    {
      return false;
    }


   private:
    /// material parameters
    Core::Mat::PAR::Parameter* params_;
  };

  /*--------------------------------------------------------------------*/
  /*! \class InelasticDefgradNoGrowth

   This class models materials in combination with the multiplicative split material that feature
   no volume changes, i.e. the inelastic deformation gradient is always the identity tensor and
   contributions to the linearizations therefore vanish.
   */
  class InelasticDefgradNoGrowth : public InelasticDefgradFactors
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params           pointer to material specific parameters
     */
    explicit InelasticDefgradNoGrowth(Core::Mat::PAR::Parameter* params);

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) override;

    PAR::InelasticSource get_inelastic_source() override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_no_growth;
    }

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    // identity tensor
    Core::LinAlg::Matrix<3, 3> identity_;
  };

  /*--------------------------------------------------------------------*/
  /*! \class InelasticDefgradTimeFunct

  This class models materials in combination with the multiplicative split material that feature
  isotropic volume changes based on a given time-dependent function for the determinant of the
  inelastic part.
  */
  class InelasticDefgradTimeFunct : public InelasticDefgradFactors
  {
   public:
    explicit InelasticDefgradTimeFunct(Core::Mat::PAR::Parameter* params);

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override {};

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override {};

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) override {};

    PAR::InelasticSource get_inelastic_source() override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_time_funct;
    }

    Mat::PAR::InelasticDefgradTimeFunct* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradTimeFunct*>(
          Mat::InelasticDefgradFactors::parameter());
    }

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    //! evaluated function value. Gets filled in pre_evaluate()
    double funct_value_;

    //! identity tensor
    Core::LinAlg::Matrix<3, 3> identity_;
  };

  class InelasticDefgradScalar : public InelasticDefgradFactors
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params           pointer to material specific parameters
     */
    explicit InelasticDefgradScalar(Core::Mat::PAR::Parameter* params);

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override = 0;

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other,
        Core::LinAlg::Matrix<3, 3>& iFinM) override = 0;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) override = 0;

    PAR::InelasticSource get_inelastic_source() override = 0;

    Core::Materials::MaterialType material_type() const override = 0;

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void set_concentration_gp(double concentration) override;

    Mat::PAR::InelasticDefgradScalar* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradScalar*>(
          Mat::InelasticDefgradFactors::parameter());
    }

    void update() override = 0;

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override = 0;

    void pack_inelastic(Core::Communication::PackBuffer& data) const override = 0;

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override = 0;

   protected:
    //! Get vector of concentration at current Gauss point
    [[nodiscard]] const std::vector<double>& get_concentration_gp() const
    {
      FOUR_C_ASSERT_ALWAYS(concentrations_ != nullptr, "Concentrations are not set");
      return *concentrations_;
    };

   private:
    /// vector of concentations at the gauss points
    std::shared_ptr<std::vector<double>> concentrations_{};
  };

  /*--------------------------------------------------------------------*/
  /*! \class InelasticDefgradPolyIntercalFrac

   This class evaluates polynomial and its first derivative w.r.t. intercalation fraction which is
   required in various routines of subclasses for isotropic and anisotropic case. This polynomial
   describes the growth of material with respect to intercalation fraction and it is prescribed by
   user in input file by defining it coefficients.
   */
  class InelasticDefgradPolyIntercalFrac : public InelasticDefgradScalar
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params             pointer to material specific parameters
     * @param[in] polynomial_growth  pointer to object that evaluates the polynomial as prescribed
     *                               in the input file
     */
    explicit InelasticDefgradPolyIntercalFrac(Core::Mat::PAR::Parameter* params);

    /*!
     * @brief evaluate polynomial describing growth of material with regard to intercalation
     * fraction based on the current concentration
     *
     * @param[in] concentration current concentration
     * @param[in] detjacobian   determinant of the deformation gradient
     * @return value of polynomial describing the growth according to current intercalation fraction
     */
    double evaluate_polynomial(double concentration, double detjacobian);

    /*!
     * @brief evaluate the first derivative of the polynomial describing the growth
     *
     * @param[in] concentration current concentration
     * @param[in] detjacobian   determinant of the deformation gradient
     * @return first derivative of the polynomial describing the growth
     */
    double evaluate_polynomial_derivative(double concentration, double detjacobian);

    Core::Materials::MaterialType material_type() const override = 0;

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other,
        Core::LinAlg::Matrix<3, 3>& iFinM) override = 0;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override = 0;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override = 0;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) override = 0;

    Mat::PAR::InelasticSource get_inelastic_source() override;

    Mat::PAR::InelasticDefgradPolyIntercalFrac* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFrac*>(
          Mat::InelasticDefgradScalar::parameter());
    }

    void update() override = 0;

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override = 0;

    void pack_inelastic(Core::Communication::PackBuffer& data) const override = 0;

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override = 0;

   private:
    /// pointer to class that evaluates the polynomial growth law
    std::shared_ptr<InelasticDefgradPolynomialShape> polynomial_growth_;
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradLinScalarIso
        This inelastic deformation gradient provides an isotropic growth law. Volumetric change due
        to this law is dependent on the current concentration \f$ c \f$ as follows :
      \f[
      \boldsymbol{F} _\text{in} = \left[1 + \text { scalar1_molar_growth_fac }
      \left(c \det \boldsymbol{F} - \text { Scalar1refconc } \right) \right] ^ { 1 / 3 }
      \boldsymbol{I}
      \f]
      */
  class InelasticDefgradLinScalarIso : public InelasticDefgradScalar
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params          pointer to material specific parameters
     * @param[in] linear_growth   pointer to object that evaluates the linear relation as prescribed
     *                            in the input file
     */
    explicit InelasticDefgradLinScalarIso(Core::Mat::PAR::Parameter* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_lin_scalar_iso;
    }

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdc) override;

    Mat::PAR::InelasticSource get_inelastic_source() override;

    Mat::PAR::InelasticDefgradLinScalar* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradLinScalar*>(
          Mat::InelasticDefgradScalar::parameter());
    }

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    /// pointer to class that evaluates the linear growth law
    std::shared_ptr<InelasticDefgradLinearShape> linear_growth_;
  };  // namespace Mat

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradLinScalarAniso

     This inelastic deformation gradient provides an anisotropic growth law.
     Volumetric change due to this law is dependent on the current concentration \f$ c \f$ as
     follows:
     \f[
     \mathbf{F}_\text{in} = \mathbf{I} + \left[ \text{scalar1_molar_growth_fac}
     \left( c \det\mathbf{F}  - \text{Scalar1refconc} \right) \right] \mathbf{G},
     \f]
     where \f$ \mathbf{G} \f$ (#growthdirmat_) is a matrix providing the information of the
     growth direction, that is constructed as follows:
     \f$ \mathbf{G} = \mathbf{g} \otimes \mathbf{g} \f$,
     where \f$ \mathbf{g} \f$ is the growth direction vector given in the input file.
     \f$ \mathbf{g} \f$ is normalized to length 1 before calculation of \f$ \mathbf{G} \f$.
     \f$ f(\chi) \f$ is defined by the user in the input file.
     */
  class InelasticDefgradLinScalarAniso : public InelasticDefgradScalar
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params          pointer to material specific parameters
     * @param[in] linear_growth   pointer to object that evaluates the linear relation as prescribed
     *                            in the input file
     */
    explicit InelasticDefgradLinScalarAniso(Core::Mat::PAR::Parameter* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_lin_scalar_aniso;
    }

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdc) override;

    Mat::PAR::InelasticSource get_inelastic_source() override;

    Mat::PAR::InelasticDefgradLinScalarAniso* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradLinScalarAniso*>(
          Mat::InelasticDefgradScalar::parameter());
    }

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    /// store pointer to class that evaluates the linear growth law
    std::shared_ptr<InelasticDefgradLinearShape> linear_growth_;
  };  // end of InelasticDefgradLinScalarAniso

  /*--------------------------------------------------------------------*/
  /*! \class InelasticDefgradPolyIntercalFracIso

   This inelastic deformation gradient provides an isotropic growth law.
   Volumetric change due to this law is non-linearly dependent on the intercalation fraction
   \f$ \chi \f$ as follows:
   \f[
   \boldsymbol{F}_\text{in} =
   \left[ \frac{f(\chi) + 1 }{f(\chi^0) + 1} \right]^{1/3} \boldsymbol{I},
   \f]
   where \f$ f(\chi) \f$ is defined by the user in the input file.
   */
  class InelasticDefgradPolyIntercalFracIso : public InelasticDefgradPolyIntercalFrac
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params             pointer to material specific parameters
     * @param[in] polynomial_growth  pointer to object that evaluates the polynomial as prescribed
     *                               in the input file
     */
    explicit InelasticDefgradPolyIntercalFracIso(Core::Mat::PAR::Parameter* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_poly_intercal_frac_iso;
    }

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdc) override;

    Mat::PAR::InelasticDefgradPolyIntercalFrac* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFrac*>(
          Mat::InelasticDefgradPolyIntercalFrac::parameter());
    }

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradPolyIntercalFracAniso

   This inelastic deformation gradient provides an anisotropic growth law.
   Volumetric change due to this law is nonlinearly dependent on the intercalation fraction
   \f$ \chi \f$ as follows:
   \f[
   \boldsymbol{F}_\text{in} =
   \boldsymbol{I} + \left[ \frac{f(\chi) - f(\chi^0)}{f(\chi^0) + 1} \right] \boldsymbol{G},
   \f]
   where \f$ \boldsymbol{G} \f$ (#growthdirmat_) is a matrix providing the information of the growth
   direction, that is constructed as follows:
   \f$ \boldsymbol{G} = \boldsymbol{g} \otimes \boldsymbol{g} \f$, where \f$ \boldsymbol{g} \f$ is
   the growth direction vector given in the input file.
   \f$ \boldsymbol{g} \f$ is normalized to length 1 before calculation of \f$ \boldsymbol{G} \f$.
   \f$ f(\chi) \f$ is defined by the user in the input file.
   */
  class InelasticDefgradPolyIntercalFracAniso : public InelasticDefgradPolyIntercalFrac
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params             pointer to material specific parameters
     * @param[in] polynomial_growth  pointer to object that evaluates the polynomial as prescribed
     *                               in the input file
     */
    explicit InelasticDefgradPolyIntercalFracAniso(Core::Mat::PAR::Parameter* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_poly_intercal_frac_aniso;
    }

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdc) override;

    Mat::PAR::InelasticDefgradPolyIntercalFracAniso* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFracAniso*>(
          Mat::InelasticDefgradPolyIntercalFrac::parameter());
    }

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradLinTempIso
   *Volumetric change due to this law depends on the temperature linearly.
   \f$ T \f$ as follows:
   \f[
   \boldsymbol{F}_\text{in} = \boldsymbol{I} \left[ 1 + \beta \left( T - T_\text{ref} \right)
   \right]^\frac{1}{3}, \f]
   */
  class InelasticDefgradLinTempIso : public InelasticDefgradFactors
  {
   public:
    explicit InelasticDefgradLinTempIso(Core::Mat::PAR::Parameter* params);

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdT) override;

    Mat::PAR::InelasticSource get_inelastic_source() override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_lin_temp_iso;
    };

    Mat::PAR::InelasticDefgradLinTempIso* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradLinTempIso*>(
          Mat::InelasticDefgradFactors::parameter());
    }

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    /// temperature at the gauss point
    double temperature_ = 0.0;
  };

  /*! \class InelasticDefgradTransvIsotropElastViscoplast
   * \brief Finite strain framework for isotropic and transversely isotropic viscoplasticity with
   * arbitrary flow rule and hardening law.
   *
   * This class implements an inelastic deformation gradient, which models viscoplastic material
   * response in a highly adaptable manner, assuming isothermal conditions at a constant
   * temperature.
   * Both isotropic and transversely isotropic material behavior can be modeled. For transversely
   * isotropic materials, both the elastic and viscoplastic deformation components can depend on the
   * preferred material fiber direction. An additive split of isotropic and transversely isotropic
   * components is assumed for the formulated elastic free energy, see Bonet et al. 1998 (below).
   * Furthermore, the model is formulated to allow for an arbitrary choice of the local viscoplastic
   * flow rule and the hardening law, see class ViscoplasticLaws. In this context, "local" refers to
   * the fact that both the flow rule and the hardening are specified independently at each Gauss
   * point, without influence from other Gauss points.
   * Currently, the model only accounts for isotropic hardening.
   * For further information on the model, refer to:
   *   -# Master's Thesis : Dragos-Corneliu Ana, Continuum Modeling and Calibration of
   * Viscoplasticity in the Context of the Lithium Anode in Solid State Batteries, Supervisor:
   * Christoph Schmidt, 2024
   *   -# Mareau et al., A thermodynamically consistent formulation of the Johnson-Cook model,
   *  Mechanics of Materials 143, 2020
   *  -# Aravas, Finite Elastoplastic Transformations of Transversely Isotropic Metals, Int. J.
   * Solids Structures Vol. 29, No. 17, 1992 (Hill 1948 yield condition used in the implemented
   * model, but with notation following Dafalias 1989, see below)
   *  -# Dafalias and Rashid, The Effect of Plastic Spin on Anisotropic Material Behavior, Int. J.
   * Plasticity, Vol. 5, 1989
   *  -# Holzapfel, Nonlinear Solid Mechanics, Wiley & Sons, 2000
   *  -# Bonet et al., A simple orthotropic, transversely isotropic hyperelastic constitutive
   * equation for large strain computations, Comput. Methods Appl. Mech. Engrg. 162, 1998
   */
  class InelasticDefgradTransvIsotropElastViscoplast : public InelasticDefgradFactors
  {
   public:
    /*!
     * @brief construct transversely isotropic material
     *
     * @param[in] params material parameters
     * @param[in] viscoplastic_law viscoplasticity law, determining the flow rule and the hardening
     *                             model
     * @param[in] fiber_reader dummy hyperelastic model utilized to read the fiber direction for
     * transverse isotropy
     * @param[in] pot_sum_el elastic components / potential summands (only isotropic)
     * @param[in] pot_sum_el_transv_iso elastic components / potential summands (only transversely
     * isotropic)
     */

    explicit InelasticDefgradTransvIsotropElastViscoplast(Core::Mat::PAR::Parameter* params,
        std::shared_ptr<Mat::Viscoplastic::Law> viscoplastic_law,
        Mat::Elastic::CoupTransverselyIsotropic fiber_reader,
        std::vector<std::shared_ptr<Mat::Elastic::Summand>> pot_sum_el,
        std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>>
            pot_sum_el_transv_iso);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_transv_isotrop_elast_viscoplast;
    }

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override {};

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdT) override {};

    Mat::PAR::InelasticSource get_inelastic_source() override { return PAR::InelasticSource::none; }

    Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast*>(
          Mat::InelasticDefgradFactors::parameter());
    }

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    /*!
     * Perform all preparation tasks for the return mapping in the current timestep. In contrast to
     * the pre_evaluate method, these tasks shall not be repeated in case of the redundant
     * evaluate call, see Issue #121 at
     * https://github.com/4C-multiphysics/4C/issues/121. This means that
     * the current, public pre-evaluate method performs only the safely
     * repeatable pre-evaluation tasks.
     * This also means that we prepare and perform the return mapping within
     * evaluate_inverse_inelastic_defgrad only if we are not in the
     * redundant call (see quick-fix PR #131 at
     * https://github.com/4C-multiphysics/4C/pull/131).
     *
     * @param[in] defgrad Deformation gradient \f$ \boldsymbol{F} \f$
     */
    void prepare_return_mapping(const Core::LinAlg::Matrix<3, 3>& defgrad);

    /*!
     * @brief Prepare LNGI -> perform all necessary preparation tasks for the evaluation of the LNGI
     * within the current global iteration, including starting point determination, reset of bounds,
     * ...
     */
    void prepare_lngi(const Core::LinAlg::Matrix<3, 3>& defgrad);

    /*!
     * @brief Compute and set LNGI starting points for the current time step
     */
    void determine_lngi_starting_points();

    /*!
     * @brief Determine the updated plastic predictor (on the yield surface) iteratively to replace
     * the initially constructed plastic predictor ; effectively reruns the pre-evaluate routine of
     * the LNGI with an updated plastic deformation gradient
     *
     *  @param[in] defgrad deformation gradient
     */
    void determine_updated_plastic_predictor_lngi(const Core::LinAlg::Matrix<3, 3>& defgrad);

    /*!
     * @brief Update data required for the next timestep LNGI at a given gp
     *
     * @param[in] gp Gauss point index
     */
    void update_lngi_data(const unsigned int gp);

    void update() override;

    void pack_inelastic(Core::Communication::PackBuffer& data) const override;

    void unpack_inelastic(Core::Communication::UnpackBuffer& buffer) override;

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

    /*! @brief Evaluate the current state variables based on a given right Cauchy-Green
     * deformation tensor, given inverse plastic deformation gradient and given equivalent
     * plastic strain
     *
     * @param[in] CM right Cauchy-Green deformation tensor \f[ \boldsymbol{C} \f] in matrix form
     * @param[in] iFinM inverse inelastic deformation gradient
     *                  \f[ \boldsymbol{F}_{\text{in}}^{-1} \f] in matrix form
     * @param[in] plastic_strain plastic strain  \f$ \varepsilon_{\text{p}} \f$
     * @param[out] err_status error status
     * @param[in] dt time step (or substep) length used for time integration
     * @param[in] eval_type evaluation type: full evaluation or only
     * partial evaluation, e.g. stop once the plastic strain rate has
     * been evaluated
     */
    StateQuantities evaluate_state_quantities(const Core::LinAlg::Matrix<3, 3>& CM,
        const Core::LinAlg::Matrix<3, 3>& iFinM, const double plastic_strain, ErrorType& err_status,
        const double dt, const StateQuantityEvalType& eval_type);

    /*! @brief Evaluate the current state variable derivatives with respect to the right
     * Cauchy-Green deformation tensor, the inverse plastic deformation gradient and the equivalent
     * plastic strain (for a given/calculated state)
     *
     * @param[in] CM right Cauchy-Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] iFinM inverse inelastic deformation gradient \f$ \boldsymbol{F}_{\text{in}}^{-1}
     *                  \f$ in matrix form
     * @param[in] plastic_strain plastic strain  \f$ \varepsilon_{\text{p}} \f$
     * @param[out] err_status error status
     * @param[in] dt time step length  \f$ \Delta t
     * \f$ (used for the integration)
     * @param[in] eval_state boolean: do we want to also evaluate the current state first (true)
     *                       or is this already available from the
     *                       current state variables (false)
     * @param[in] eval_type evaluation type: full evaluation or only
     * partial evaluation, e.g. stop once the derivatives of the plastic strain rate have
     * been evaluated
     */
    StateQuantityDerivatives evaluate_state_quantity_derivatives(
        const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<3, 3>& iFinM,
        const double plastic_strain, ErrorType& err_status, const double dt,
        const StateQuantityDerivEvalType& eval_type, const bool eval_state = false);

    //! return the fiber direction of transverse isotropy for the considered element
    Core::LinAlg::Matrix<3, 1> get_fiber_direction() { return m_; }

    /*!
     * @brief Set the last_ time step quantities of the material at a
     * specified GP. To be
     * used during the debugging of the time integration algorithm.
     * @note to be used only for debugging purposes!
     */
    void debug_set_last_quantities(const int gp,
        const Core::LinAlg::Matrix<3, 3>& last_plastic_defgrad_inverse,
        const double last_plastic_strain, const double last_plastic_strain_increment,
        const double last_equiv_stress, const double last_equiv_stress_elastic_pred,
        const double last_equiv_stress_plastic_pred, const Core::LinAlg::Matrix<3, 3>& last_defgrad,
        const Core::LinAlg::Matrix<3, 3>& last_rightCG, const double last_xi_lambda_1,
        const double last_xi_lambda_2, const std::array<double, 3> last_xi_eigenvect_rot,
        const double last_max_xi_lambda_1, const double last_max_xi_lambda_2,
        const std::array<double, 3> last_max_xi_eigenvect_rot, const double optimal_xi_lambda_1,
        const double optimal_xi_lambda_2, const std::array<double, 3> optimal_xi_eigenvect_rot);

    /*!
     * @brief Get the utilized viscoplastic law object.
     * @note to be used only for debugging purposes!
     */
    std::shared_ptr<Mat::Viscoplastic::Law> debug_get_viscoplastic_law()
    {
      return viscoplastic_law_;
    };

    /*!
     * @brief Set the flag for updating history variables.
     * @note to be used only for debugging purposes!
     */
    void debug_set_update_hist_var(const bool update_hist_var)
    {
      update_hist_var_ = update_hist_var;
    }

   private:
    //! constant material tensors     (isotropic: constant tensors
    //! such
    //! as identity matrices; transversely-isotropic: also contains
    //! tensors associated with the director vector)
    ConstMatTensors const_mat_tensors_;

    //! current Gauss Point
    int gp_;
    //! current element ID
    int ele_gid_;

    //! parameter list
    Teuchos::ParameterList params_;


    //! map to elastic materials/potential summands (only isotropic)
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsumel_;

    //! map to elastic materials/potential summands (only transversely isotropic)
    std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> potsumel_transviso_;

    //! viscoplastic law
    std::shared_ptr<Mat::Viscoplastic::Law> viscoplastic_law_;

    //! fiber reader (hyperelastic transversely isotropic material used for fiber reading)
    Mat::Elastic::CoupTransverselyIsotropic fiber_reader_;

    //! fiber direction (director vector)
    Core::LinAlg::Matrix<3, 1> m_;

    //! utilities for evaluating the matrix exponential and logarithm
    MatrixExpLogUtils matrix_exp_log_utils_;

    //! boolean to control whether the history variables should be updated during evaluation
    bool update_hist_var_ = true;

    //! boolean to control whether to use the elastic predictor directly or to use LNGI (in cases
    //! where performing the LNGI is ineffective, e.g., if the elastic predictor already leads to a
    //! very small plastic strain increment)
    bool use_elastic_predictor_ = false;

    //! control variable: should the Local Newton Guess Interpolation compute its starting points
    //! within the current time step? -> we want to do this only once for all GP for the current
    //! starting point choices, since these computations can get expensive
    bool compute_lngi_starting_points_ = false;

    //! tracker for time step settings and time instants
    TimeStepTracker time_step_tracker_;

    //! tracker for quantities at the last and current time points (i.e., at \f[ t_n \f] and
    //! \f[ t_{n+1} \f], respectively) for all Gauss points simultaneously
    TimeStepQuantities time_step_quantities_;

    //! evaluated state quantities
    StateQuantities state_quantities_;

    //! evaluated state quantity derivatives
    StateQuantityDerivatives state_quantity_derivatives_;

    //! tensor interpolator used in the substepping procedure (one-dimensional, of order 1)
    Core::LinAlg::SecondOrderTensorInterpolator<1> tensor_interpolator_;

    //! tensor interpolation: reference matrices
    std::vector<Core::LinAlg::Matrix<3, 3>> ref_matrices_;

    //! tensor interpolation: 1D reference locations (we always interpolate between 0.0 and 1.0
    //! based on the reference matrices of the current time step)
    const std::vector<double> ref_locs_{0.0, 1.0};

    //! tracker object for the Local Newton initial guess interpolation
    LocalNewtonGuessInterpolation lnl_guess_interpolation_;

    //! tracker for current global iteration index
    unsigned int globiter_;

    //! micro iteration data for all microiterations
    //! of the Local Newton Guess Interpolation, to be written to csv
    CSVOutputPredAdaptMicroIterData csv_output_lngi_micro_iter_data_;

    //! micro iteration data for all microiterations
    //! of the line search, to be written to csv
    CSVOutputLineSearchMicroIterData csv_output_line_search_micro_iter_data_;

    //! tracker object for the local substepping procedure
    LocalSubsteppingUtils local_substepping_utils_;

    //! tracking object for Local Newton data
    mutable LocalNewtonData lnl_data_;

    /*!
     * @brief Calculate the Holzapfel gamma and delta values of the isotropic elastic material
     * components
     * @param[in] CeM elastic right Cauchy_Green deformation tensor \f$ \boldsymbol{C}_\text{e}
     * \f$ in matrix form
     * @param[out] gamma stress factors for the isotropic elasticity case, as derived in
     *                   Holzapfel - Nonlinear Solid Mechanics(2000)
     * @param[out] delta constitutive tensor factors for the isotropic elasticity case, as derived
     *                   in Holzapfel - Nonlinear Solid Mechanics(2000)
     */
    void calculate_gamma_delta(const Core::LinAlg::Matrix<3, 3>& CeM,
        Core::LinAlg::Matrix<3, 1>& gamma, Core::LinAlg::Matrix<8, 1>& delta);

    /*!
     * @brief Check if the elastic predictor provides the solution for the current time step,
     * i.e., the deformation in the current time step is purely elastic with no viscoplastic
     * contribution.
     *
     * @param[in] CM right Cauchy_Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] iFinM_pred predictor of the inverse inelastic deformation gradient \f$
     * \bm{F}_{\text{in, pred}} \f$
     * @param[in] plastic_strain_pred predictor of the plastic strain \f$ \varepsilon_{\text{p,
     * pred}} \f$
     * @param[out] err_status error status
     * @return boolean value: true (predictor = solution), or false (predictor != solution)
     */
    bool check_elastic_predictor(const Core::LinAlg::Matrix<3, 3>& CM,
        const Core::LinAlg::Matrix<3, 3>& iFinM_pred, const double plastic_strain_pred,
        ErrorType& err_status);

    /*!
     * @brief Calculate the residual for the Local Newton Loop (LNL)
     *
     * @note The state quantities are updated in this method, since they
     * are used for the computation of the residual!
     *
     * @param[in] CM right Cauchy_Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] x vector of Local Newton Loop unknowns, composed of the components of the
     * inverse inelastic deformation gradient \f$ \boldsymbol{F}_{\text{in}}^{-1} \f$ and plastic
     * strain \f$ \varepsilon_{\text{p}} \f$
     * @param[in] last_iFpM last inverse plastic deformation gradient
     *                      \f$ \boldsymbol{F}_{\text{in}, n}^{-1} \f$ in matrix form
     * @param[in] last_plastic_strain last plastic strain \f$ \varepsilon_{\text{p}, n}\f$
     * @param[in] dt time step (or substep) length used for time integration
     * @param[out] err_status error status
     * @return  residual of the LNL equations
     */
    Core::LinAlg::Matrix<10, 1> calculate_local_newton_loop_residual(
        const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<10, 1>& x,
        const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double last_plastic_strain,
        const double dt, ErrorType& err_status);


    /*!
     * @brief For a given right Cauchy_Green tensor and the Local NR Loop unknown vector,
     * compute the 10 x 10 Jacobian matrix required for the Local Newton Loop and the
     * linearization for the Global Newton Loop
     *
     * @note The state quantity derivatives are updated in this method.
     * They require the state quantities, which were evaluated and stored
     * previously when calculating the residual.
     *
     * @param[in] CM right Cauchy_Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] x vector of Local Newton Loop unknowns, composed of the components of the
     * inverse inelastic deformation gradient \f$ \boldsymbol{F}_{\text{in}}^{-1} \f$ and plastic
     * strain \f$ \varepsilon_{\text{p}} \f$
     * @param[in] last_iFpM last inverse plastic deformation gradient
     *                      \f$ \boldsymbol{F}_{\text{in}, n}^{-1} \f$ in matrix form
     * @param[in] last_plastic_strain last plastic strain \f$ \varepsilon_{\text{p}, n}\f$
     * @param[in] dt time step (or substep) length used for time integration
     * @param[out] err_status error status
     * @return 10x10 jacobian matrix of the Local Newton Loop and of the linearization
     *         \f$ \boldsymbol{J} \f$
     */
    Core::LinAlg::Matrix<10, 10> calculate_jacobian(const Core::LinAlg::Matrix<3, 3>& CM,
        const Core::LinAlg::Matrix<10, 1>& x, const Core::LinAlg::Matrix<3, 3>& last_iFinM,
        const double last_plastic_strain, const double dt, ErrorType& err_status);

    /*!
     * @brief Interpolate a valid initial guess for the Local Newton Loop,
     * provided a current initial guess
     *
     * @note If the current initial guess is a valid initial guess (i.e.,
     * numerically evaluable and leads to plastic flow), then it is directly
     * return without further interpolation
     * @param[in] FM deformation gradient
     * @return interpolated initial guess with the same structure as the current
     * initial guess
     */
    Core::LinAlg::Matrix<10, 1> interpolate_local_newton_guess(
        const Core::LinAlg::Matrix<3, 3>& FM);

    /*!
     * @brief Local Newton Loop in order to calculate the current inverse plastic deformation
     * gradient and the current plastic strain value
     *
     * @param[in] defgrad deformation gradient \f$ \boldsymbol{F} \f$ in matrix form
     * @param[in] x initial guess of Local Newton Loop, composed of the components of the
     *              inverse inelastic deformation gradient \f$ \boldsymbol{F}_{\text{in}}^{-1} \f$
     *              and plastic strain \f$ \varepsilon_{\text{p}} \f$
     * @param[out] err_status error status
     * @return solution vector of the Local Newton Loop, structured analogously to the initial guess
     * x
     */
    Core::LinAlg::Matrix<10, 1> local_newton_loop(const Core::LinAlg::Matrix<3, 3>& defgrad,
        const Core::LinAlg::Matrix<10, 1>& x, ErrorType& err_status);


    /*!
     * @brief Performs return mapping at each GP. It first evaluates whether the elastic predictor
     * is suitable as a solution, and performs the local time integration (Local Newton Loop)
     * afterwards.
     *
     * @param[in] FredM reduced deformation gradient \f$ \boldsymbol{F}_{\text{red}} =
     * \boldsymbol{F} \boldsymbol{F_{\text{in,other}}^{-1}} \f$ accounting for all the already
     * computed inelastic defgrad factors
     * @return inverse inelastic deformation gradient \boldsymbol{F}_{\text{in}}^{-1}
     */
    Core::LinAlg::Matrix<3, 3> return_mapping(const Core::LinAlg::Matrix<3, 3>& FredM);


    /*!
     * @brief Compute the plastic strain \f$
     * \varepsilon^{\text{p}}_{n+1} \f$, given the equivalent stress \f$
     * \overline{\sigma} \f$.
     *
     * The computation is performed using the discretized evolution
     * equation (Backward Euler):
     * \f$ \varepsilon^{\text{p}}_{n+1} =  \varepsilon^{\text{p}}_{n} +
     * \Delta t v^{\text{p}}(\overline{sigma}_{n+1},
     * \varepsilon^{\text{p}}_{n+1}) \f$, where \f$
     * v^{\text{p}}(\overline{\sigma}, \varepsilon^{\text{p}})  \f$
     * is characteristic to the employed flow rule.
     *
     *
     * @param[in] equiv_stress equivalent stress \f$ \overline{\sigma}_{n+1} \f$
     * @param[in] last_plastic_strain plastic strain at the last time instant \f$
     * \varepsilon^{\text{p}}_{n} \f$
     * @param[in] dt time step \f$ \Delta t \f$
     * @param[out] err_status error status
     * @return plastic strain \f$ \varepsilon^{\text{p}}_{n+1} \f$
     */
    double integrate_plastic_strain(const double equiv_stress, const double last_plastic_strain,
        const double dt, ErrorType& err_status);

    /*!
     * @brief Evaluate whether this is a valid initial guess for the Local
     * Newton scheme: numerically
     * evaluable residual and Jacobian, and leading to plastic flow
     *
     * @param[in] defgrad Deformation gradient
     * @param[in] inv_defgrad Inverse deformation gradient
     * @param[in] right_cg_tensor Right Cauchy-Green deformation tensor
     * @param[in] inv_plastic_defgrad_guess Inverse plastic
     * deformation gradient to be verified as initial guess
     * @param[in] plastic_strain_guess Inverse plastic
     * strain to be verified as initial guess
     * @param[out] err_status Error type obtained during verification
     * @param[out] state_quantities State quantities obtained during
     * verification
     * @param[out] state_quantity_derivatives State quantity derivatives obtained during
     * verification
     */
    void is_valid_local_newton_initial_guess(const Core::LinAlg::Matrix<3, 3>& defgrad,
        const Core::LinAlg::Matrix<3, 3>& inv_defgrad,
        const Core::LinAlg::Matrix<3, 3>& right_cg_tensor,
        const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_guess,
        const double plastic_strain_guess, ErrorType& err_status, StateQuantities& state_quantities,
        StateQuantityDerivatives& state_quantity_derivatives);

    /*!
     * @brief Get the line search step size  for the current iteration of
     * the Local Newton Loop
     *
     * @note During the iterations of the Local Newton Loop, the plastic
     * strain may be updated such that it becomes negative, which is both
     * nonphysical and problematic in the computation of certain
     * viscoplasticity flow rules and/or hardening models. To address
     * this, we compute a line search parameter $\alpha_i$
     * to update the solution f$ \boldsymbol{s}_{i+1} =
     * \boldsymbol{s}_{i} + \alpha_i \Delta \boldsymbol{s}_{i+1} \f$
     * such that the negative plastic strain is limited to positive
     * values. Analogously, we account for other possible errors, such
     * as e.g, overflow. For the inexact line search, we use the backtracking
     * algorithm as presented in:
     *
     * -# Andrei 2022, Modern Numerical Nonlinear Optimization, Vol.
     * 195, Springer Optimization and its Applications, DOI:
     * 10.1007/978-3-031-08720-2
     *
     * @param[in] curr_sol solution of the current iteration of the
     * Local Newton Loop \f$ \boldsymbol{s}_i \f$
     * @param[in] CM right Cauchy_Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] curr_res residual of the current iteration of the
     * Local Newton Loop \f$ \boldsymbol{r}_{\boldsymbol{s}_i} \f$
     * @param[in] incr increment \f$ \Delta \boldsymbol{s}_{i+1} \f$ for
     * the update of the solution vector
     * @param[out] err_status error status
     * @return line search step \f$ \alpha \f$
     *
     */
    double get_line_search_step(const Core::LinAlg::Matrix<10, 1>& curr_sol,
        const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<10, 1>& curr_res,
        const Core::LinAlg::Matrix<10, 1>& incr, ErrorType& err_status);


    /*!
     * @brief Setup new substep in the Local Newton Loop in case of an encountered evaluation
     * error
     *
     * @param[in,out] sol current solution vector of the Local Newton Loop (reset to the last
     * converged value within this method)
     * @param[in,out] curr_CM current right Cauchy-Green deformation tensor, interpolated using
     * the reference matrices of the time step (interpolated again within this method with the
     * updated new substep length)
     * @return error status for the new substep (true: no errors, false: we have halved the time
     * step too many times)
     *
     */
    bool prepare_new_substep(Core::LinAlg::Matrix<10, 1>& sol, Core::LinAlg::Matrix<3, 3>& curr_CM);

    /*!
     * @brief Routine utilized during the Local Newton Loop
     * evaluations. The performed steps depend on the input error status
     * and the user settings (e.g. substepping, Local Newton Guess
     * Reinterpolations, ...).
     *
     *
     * @param[in] err_status error status
     * @param[in,out] sol current solution vector of the Local Newton Loop (reset to the last
     * converged value within this method)
     * @param[in,out] curr_CM current right Cauchy-Green deformation tensor, interpolated using
     * the reference matrices of the time step (interpolated again within this method with the
     * updated new substep length)
     * @return action to be performed subsequently in the LNL
     */
    ErrorAction manage_evaluation_error(const ErrorType& err_status,
        Core::LinAlg::Matrix<10, 1>& sol, Core::LinAlg::Matrix<3, 3>& curr_CM);

    /*!
     * @brief Evaluate the additional cmat stiffness tensor using a perturbation-based approach,
     if
     * the analytical evaluation fails
     *
     * @note For further information on the procedure, refer to:
     *       -# Master's Thesis : Dragos-Corneliu Ana, Continuum Modeling and Calibration of
     * Viscoplasticity in the Context of the Lithium Anode in Solid State Batteries, Supervisor:
     *
     Christoph Schmidt, 2024
     *
     * @param[in] FredM reduced deformation gradient \f$ \boldsymbol{F}_{\text{red}} =
     * \boldsymbol{F} \boldsymbol{F_{\text{in,other}}^{-1}} \f$ accounting for all the already
     * computed inelastic defgrad factors
     * @param[out] cmatadd Additional elasticity stiffness
     * @param[in] iFin_other Already computed inverse inelastic deformation gradient
     *              (from already computed inelastic factors in the multiplicative split material)
     * @param[in] dSdiFinj Derivative of 2nd Piola Kirchhoff stresses w.r.t. the inverse inelastic
     *                     deformation gradient of current inelastic contribution
     *
     */
    void evaluate_additional_cmat_perturb_based(const Core::LinAlg::Matrix<3, 3>& FredM,
        Core::LinAlg::Matrix<6, 6>& cmatadd, const Core::LinAlg::Matrix<3, 3>& iFin_other,
        const Core::LinAlg::Matrix<6, 9>& dSdiFinj);

    /*!
     * @brief Get an extensive error message to be displayed when the
     * simulation terminates. This is used for debugging the time
     * integration in more detail. This message contains a base error
     * message which describes what failed in a short form - this is
     * then extended with information on the element ID, the Gauss
     * Point, the last_ values and so on...
     *
     * @param[in] base_error_string base error message to be extended
     * with further information
     */
    std::string debug_get_error_info(const std::string& base_error_string);


    // benchmarking procedure: runs a specific function in a loop until the
    // computation time converges based on a specified relative tolerance
    template <typename Func, typename... Args>
    double benchmark_function(std::string func_descr, Teuchos::Time& func_timer,
        const double relative_tol, bool& increment_timint_analysis_vars, int& num_of_required_iters,
        Func&& func, Args&&... args)
    {
      // average computation time (current iteration)
      double avg_time = 0.0;

      // average computation time (previous iteration)
      double prev_avg_time = 0.0;

      // number of performed iterations / repetitions
      num_of_required_iters = 0;

      // minimum and maximum numbers of iterations
      constexpr int warmup_iters = 3;    // number of warm-up iterations
      constexpr int max_iters = 100000;  // safety cap

      // start timer
      func_timer.start(true);

      // loop over iterations
      while (true)
      {
        // increment iterations and check safety cap
        ++num_of_required_iters;
        FOUR_C_ASSERT_ALWAYS(num_of_required_iters < max_iters,
            "Maximum number of repetitions {} was reached without a converged computation time for "
            "the function [{}]",
            max_iters, func_descr);

        // set tracker variable for iters, steps, errors, ...
        increment_timint_analysis_vars =
            (num_of_required_iters ==
                1);  // only track iters, steps, errors, ... for
                     // the first repetition / iteration of the procedure to be benchmarked


        // reset timer upon reaching minimum number of iterations (warm-up
        // iterations)
        if (num_of_required_iters == warmup_iters)
        {
          func_timer.reset();
          continue;
        }

        // run function to be timed
        func(std::forward<Args>(args)...);

        // if this is not a warm-up iteration anymore, we calculate
        // relative change and check for convergence
        if (num_of_required_iters > warmup_iters)
        {
          // get current elapsed time
          const double t = func_timer.totalElapsedTime(true);

          // running average
          avg_time = t / (num_of_required_iters - warmup_iters);

          // check for convergence based on the relative tolerance
          const double rel_change = std::abs(avg_time - prev_avg_time) / avg_time;


          // if convergence is reached: stop the timer and break out of the loop
          if (rel_change < relative_tol)
          {
            func_timer.stop();
            break;
          }

          // set previous times for the next iteration
          prev_avg_time = avg_time;
        }
      }
      // set control variable to true, since we exit the benchmarking procedure
      increment_timint_analysis_vars = true;

      // return average time
      return avg_time;
    }
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
