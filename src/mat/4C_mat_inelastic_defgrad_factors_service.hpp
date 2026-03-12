// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP

#include "4C_config.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /// namespace: utilities for
  /// InelasticDefgradTransvIsotropElastViscoplast
  namespace InelasticDefgradTransvIsotropElastViscoplastUtils
  {
    /// enum class for error types in InelasticDefgradTransvIsotropElastViscoplast, used for
    /// triggering different procedures (e.g. Reinterpolation,
    /// substepping, line search) during the
    /// Local Newton Loop
    enum class ErrorType
    {
      no_errors,                ///< no errors
      negative_plastic_strain,  ///< negative plastic strain which does not allow for evaluations
                                ///< inside the viscoplasticity laws
      overflow_error,  ///< overflow error of the term \f$ \Delta t \dot{\varepsilon}^{\text{p}} \f$
                       ///< (and \f$ \mathsymbol{E}^{\text{p}}  = \exp(- \Delta t
                       ///< \dot{\varepsilon}^{\text{p}} \mathsymbol{N}^{\text{p}}) \f$)
      no_flow_resistance,            ///< the material has no flow resistance anymore, such that the
                                     ///< evaluations model non-physical phenomena
      no_plastic_incompressibility,  ///< no plastic incompressibility, meaning that the determinant
                                     ///< of the inelastic defgrad is far from 1
      failed_solution_linear_system_lnl,  ///< solution of the linear system in the Local
                                          ///< Newton-Raphson Loop failed
      failed_determ_line_search_step,     ///< the computation of a suitable line search step failed
      no_convergence_local_newton,  ///< the Local Newton Loop did not converge for the given loop
                                    ///< settings
      singular_jacobian,  ///< singular Jacobian after converged LNL, which does not enable our
                          ///< analytical evaluation of the linearization
      failed_solution_analytic_linearization,  ///< solution of the linear system in the analytical
                                               ///< linearization failed
      failed_computation_flow_resistance,  ///< failed in the computation of the flow resistance via
                                           ///< time integration of the hardening-rate equation
                                           ///(e.g., when using the Anand law)
      failed_computation_flow_resistance_derivs,  ///< failed in the computation of the flow
                                                  ///< resistance derivatives (e.g., when using the
                                                  ///< Anand law)
      failed_matrix_log_evaluation,   ///< failed evaluation of the matrix logarithm or its
                                      ///< derivative
      failed_matrix_exp_evaluation,   ///< failed evaluation of the matrix exponential or its
                                      ///< derivative
      failed_right_cg_interpolation,  ///< failed interpolation of the right Cauchy-Green tensor
      under_yield_surface  ///< mechanical state is "under" the yield surface, i.e., the evaluated
                           ///< stress is smaller than the yield stress, which should not occur
                           ///< in the Local Newton loop
    };


    /// enum class for error management actions in the iterations of the
    // Local Newton loop
    enum class ErrorAction
    {
      continue_iteration,           ///< continue iteration without any errors (NoErrors)
      return_solution_with_errors,  ///< return the current solution with errors (if the current
                                    ///< simulation settings cannot lead to a solution)
      next_iteration,               ///< go to next iteration after performing certain reset steps
    };

    /// convert error type to detailed error message
    std::string get_detailed_error_message_for_error_type(ErrorType err_type);

    /// enum class for material behavior types
    enum class MatBehavior
    {
      isotrop,         ///< isotropic material behavior
      transv_isotrop,  ///< transversely isotropic material behavior
    };

    /// enum class for time integration types (Local Newton integration)
    enum class TimIntType
    {
      standard,     ///< standard time integration,
      logarithmic,  ///< time integration with logarithmically transformed residual equation for the
                    ///< evolution of the plastic deformation gradient
    };

    /// enum class for material linearization types
    enum class LinearizationType
    {
      analytic,  ///< analytical linearization involving the solution of a linear system of
                 ///< equations,
      perturbation_based,  ///< linearization based on perturbing the current state
    };

    //! matrix exponential and logarithm evaluation utilities
    struct MatrixExpLogUtils
    {
      //! Pade approximation order (to be used consistently: the
      //! derivative of the matrix functions should use the same Pade
      //! order as the evaluation of the matrix functions)
      unsigned int pade_order_ = 16;  // by default we set the highest order currently implemented
    };

    //! struct containing time step settings and time trackers
    struct TimeStepTracker
    {
      //! time step length
      double dt_;
      //! currently computed time instant \f$ t_{n+1} \f$
      double tnp_;
      //! minimum substep length
      double min_dt_;
    };


    //! struct containing quantities at the last and current time points (i.e., at \f[ t_n \f] and
    //! \f[ t_{n+1} \f], respectively). The quantities are tracked at all Gauss points, in order to
    //! update them simultaneously during the update method call
    struct TimeStepQuantities
    {
      //! right Cauchy-Green deformation tensor at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_rightCG_;

      //! inverse plastic deformation gradient at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_plastic_defgrad_inverse_;

      //! elastic stretch eigenvalues (ordered from
      //! highest to smallest) at the
      //! last time step (for all Gauss points)
      std::vector<std::array<double, 3>> last_elastic_stretch_eigenval_;

      //! (equivalent) plastic strain at the last time step (for all Gauss points)
      std::vector<double> last_plastic_strain_;

      //! equivalent stress at the previous time instant (for all Gauss points)
      std::vector<double> last_equiv_stress_;

      //! equivalent stress at the previous time instant (for all Gauss points) for the elastic
      //! predictor
      std::vector<double> last_equiv_stress_elastic_pred_;

      //! equivalent stress at the previous time instant (for all Gauss points) for the plastic
      //! predictor
      std::vector<double> last_equiv_stress_plastic_pred_;

      //! plastic strain increment (plastic strain rate * time step) at the previous time instant
      //! (for all Gauss points)
      std::vector<double> last_plastic_strain_increment_;

      //! last (reduced) deformation gradient: used to in the Local Newton Guess
      //! Interpolation routine
      std::vector<Core::LinAlg::Matrix<3, 3>> last_defgrad_;

      //! temporary variable, for which we store the right Cauchy-Green deformation tensor at each
      //! evaluation (used in order to update last_rightCG_ once outer NR converges) (for all Gauss
      //! points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_rightCG_;

      //! current (reduced) deformation gradient: used to check whether the inverse inelastic
      //! deformation gradient has already been evaluated (to improve the computation performance)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_defgrad_;


      //! current inverse plastic deformation gradient (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_plastic_defgrad_inverse_;

      //! current plastic strain (for all Gauss points)
      std::vector<double> current_plastic_strain_;

      //! current equivalent stress (for all Gauss points)
      std::vector<double> current_equiv_stress_;

      //! inverse plastic deformation gradient at the last computed time instant (after the last
      //! converged substep)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_substep_plastic_defgrad_inverse_;
      //! plastic strain at the last computed time instant (after the last converged substep)
      std::vector<double> last_substep_plastic_strain_;
    };



    /// struct: constant non-material tensors, such as different
    /// identity tensors
    struct ConstNonMatTensors
    {
      static const ConstNonMatTensors& instance()
      {
        static ConstNonMatTensors instance;
        return instance;
      }

      // constructor
      ConstNonMatTensors();
      // second-order 3x3 identity tensor in matrix form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<3, 3> id3x3_{Core::LinAlg::Initialization::zero};
      // second-order 3x3 identity in Voigt stress form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 1> id6x1_{Core::LinAlg::Initialization::zero};
      // symmetric identity four tensor of dimension 3 \f$ \mathbb{I}_\text{S} \f$
      Core::LinAlg::Matrix<6, 6> id4_6x6_{Core::LinAlg::Initialization::zero};
      // deviatoric operator \f$ \mathbb{P}_{\text{dev}}  =  \mathbb{I}_\text{S} -
      // \frac{1}{3} \boldsymbol{I} \otimes \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 6> dev_op_{Core::LinAlg::Initialization::zero};
      // identity fourth-order tensor in Voigt notation: delta_AC delta_BD in index notation
      Core::LinAlg::Matrix<9, 9> id4_9x9_{Core::LinAlg::Initialization::zero};
      // second-order 10x10 identity tensor in matrix form
      Core::LinAlg::Matrix<10, 10> id10x10_{Core::LinAlg::Initialization::zero};
    };



    //! struct containing constant tensors which depend on the constant fiber direction \f$
    //! \boldsymbol{m} \f$
    struct ConstMatTensors
    {
      //! \f$ \boldsymbol{I} + \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> id_plus_mm_;
      //! \f$ \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> mm_{Core::LinAlg::Initialization::zero};
      //! deviatoric part \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //! \right)_\text{dev}\f$
      Core::LinAlg::Matrix<3, 3> mm_dev_{Core::LinAlg::Initialization::zero};
      //! \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right) \otimes \left( \boldsymbol{m}
      //! \otimes \boldsymbol{m} \right) \f$ (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dyad_mm_{Core::LinAlg::Initialization::zero};
      //!  \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right)_\text{dev} \otimes \left(
      //!  \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dev_dyad_mm_{Core::LinAlg::Initialization::zero};
      //!  \f$ \boldsymbol{I} \otimes \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> id_dyad_mm_;

      //! set tensors for a given fiber direction \f$ \boldsymbol{m} \f$
      void set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m);
    };

    //! class containing utilities for initial guess interpolation in the Local
    // Newton scheme for InelasticDefgradTransvIsotropElastViscoplast
    class LocalNewtonGuessInterpolation
    {
     public:
      //! plastic predictor: types of elastic stretch eigenvalues
      enum class PlasticPredictorElasticStretchEigenvalType
      {
        maintain,   ///< elastic deformation gradient maintains its elastic stretch eigenvalues
                    ///< from the previous time instant
        eliminate,  ///< no elastic stretch: elastic stretch = unit tensor
                    ///< ; motivated by stress relaxation (elastic deformation
                    ///< -> plastic deformation) -> only completely consistent for
                    ///< no-yield-surface viscoplastic models, but numerically comfortable for
                    ///< all formulations in general
      };

      //! plastic predictor: types of elastic stretch eigenvector rotations
      enum class PlasticPredictorElasticStretchEigenvectRotType
      {
        elastic_predictor,  ///< eigenvector rotation is taken from the trial elastic state / the
                            ///< elastic predictor (for isotropic materials this is generally
                            ///< consistent with the solution of the Local Newton)
      };



      //! plastic predictor rotation assignment types (initial guess interpolation)
      enum class PlasticPredictorRotationType
      {
        elastic_predictor,  ///< elastic rotation = trial elastic rotation (elastic
                            ///< predictor)
                            /// within the plastic predictor (for isotropic models this is
                            /// generally consistent with the solution of the Local Newton)
      };



      //! starting point type (initial guess interpolation)
      enum class LocalNewtonGuessInterpolationStartingPointType
      {
        user_set,                     ///< User-set constant factor
        last_interpolation_point,     ///< Takes the interpolation point of the previous timestep,
                                      ///< which led to a valid initial guess, as the starting
                                      ///< point for the interpolation within the current timestep
        optimal_interpolation_point,  ///< Takes the optimal interpolation point
                                      ///< of the previous timestep, i.e., the interpolation
                                      ///< point leading to the solution of the Local Newton
                                      ///< loop of the last global Newton iteration, as the
                                      ///< starting point for th interpolation within the
                                      ///< current timestep
        optimal_equiv_stress          ///< Similar to optimal_interpolation_point, but
                                      ///< considers the interpolation factor of the equivalent
        ///< stress for the previous timestep with respect to the previous
        ///< elastic and plastic predictors. Hence, all components of the interpolation point
        ///< are effectively set to this one factor, instead of the generally different
        ///< interpolation components obtained with optimal_interpolation_point
      };



      //! struct: components of deformation gradient (standard deformation gradient | elastic
      //! deformation gradient | plastic deformation gradient) within elastic and plastic
      //! predictors extracted from combined spectral-polar decomposition as in Satheesh et al.
      //! 2023 (10.1002/nme.7373)
      struct PredictorDefgradDecomposition
      {
        //! elastic predictor: full, non-decomposed specific deformation
        // gradient considered
        Core::LinAlg::Matrix<3, 3> specific_defgrad_elast_pred_;
        //! plastic predictor: full, non-decomposed specific deformation
        // gradient considered
        Core::LinAlg::Matrix<3, 3> specific_defgrad_plast_pred_;
        //! elastic predictor: eigenvalues \f$ \lambda_{\mathrm{elast}, i} \f$
        std::array<double, 3> lambda_elast_pred_;
        //! plastic predictor: eigenvalues \f$ \lambda_{\mathrm{plast}, i} \f$
        std::array<double, 3> lambda_plast_pred_;
        //! elastic predictor: logarithm of eigenvalues \f$ \log(\lambda_{\mathrm{elast}, i})
        //! \f$
        std::array<double, 3> log_lambda_elast_pred_;
        //! plastic predictor: logarithm eigenvalues \f$ \log(\lambda_{\mathrm{plast}, i}) \f$
        std::array<double, 3> log_lambda_plast_pred_;
        //! elastic predictor: eigenvector rotation matrix \f$ \mathbf{Q}_{\mathrm{elast}} \f$
        Core::LinAlg::Matrix<3, 3> Qmat_elast_pred_;
        //! plastic predictor: eigenvector rotation matrix \f$ \mathbf{Q}_{\mathrm{plast}} \f$
        Core::LinAlg::Matrix<3, 3> Qmat_plast_pred_;
        //! plastic predictor: relative eigenvector rotation matrix \f$
        //! \mathbf{Q}_{\mathrm{plast, rel}} \f$ with respect to the eigenvector rotation of the
        //! elastic predictor
        Core::LinAlg::Matrix<3, 3> Qmat_plast_pred_rel_;
        //! plastic predictor: relative eigenvector rotation vector \f$
        //! \mathbf{q}_{\mathrm{plast, rel}} \f$ with respect to the eigenvector rotation of the
        //! elastic predictor
        Core::LinAlg::Matrix<3, 1> Qvec_plast_pred_rel_;
        //! elastic predictor: rotation matrix \f$ \mathbf{R}_{\mathrm{elast}} \f$
        Core::LinAlg::Matrix<3, 3> Rmat_elast_pred_;
        //! plastic predictor: rotation matrix \f$ \mathbf{R}_{\mathrm{plast}} \f$
        Core::LinAlg::Matrix<3, 3> Rmat_plast_pred_;
        //! plastic predictor: relative rotation matrix \f$ \mathbf{R}_{\mathrm{plast, rel}} \f$
        //! with respect to rotation of the elastic predictor
        Core::LinAlg::Matrix<3, 3> Rmat_plast_pred_rel_;
        //! plastic predictor: relative rotation vector \f$ \mathbf{r}_{\mathrm{plast, rel}} \f$
        //! with respect to rotation of the elastic predictor
        Core::LinAlg::Matrix<3, 1> Rvec_plast_pred_rel_;
        //! elastic predictor: spectral pairs containing the eigenvectors and eigenvalues
        std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_elast_pred_;
        //! plastic predictor: spectral pairs containing the eigenvectors and eigenvalues
        std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_plast_pred_;

        /**
         * @brief Constructor.
         *
         * @note The specific deformation gradient within the elastic predictor
         * is considered as the reference when aligning eigenpairs, and
         * determining the relative rotation.
         *
         * @param[in] defgrad_elast_pred Deformation gradient (
         * elastic, inverse plastic, ...) within the elastic predictor
         * @param[in] defgrad_plast_pred Deformation gradient (elastic,
         * inverse plastic, ...) within the plastic predictor
         * @param[in] spectral_pairs_ref Given reference / elastic predictor spectral pairs to
         * be used for determining relative rotation and aligning eigenpairs for
         * the plastic predictor. Only to be used in special cases, such as when
         * we
         * determine optimal interpolation factors to ensure consistent
         * reference spectral pairs for interpolation and solution.
         */
        PredictorDefgradDecomposition(const Core::LinAlg::Matrix<3, 3>& defgrad_elast_pred,
            const Core::LinAlg::Matrix<3, 3>& defgrad_plast_pred,
            std::optional<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>>
                spectral_pairs_ref = std::nullopt);

        //! pack method
        void pack(Core::Communication::PackBuffer& data) const
        {
          Core::Communication::add_to_pack(data, specific_defgrad_elast_pred_);
          Core::Communication::add_to_pack(data, specific_defgrad_plast_pred_);
          Core::Communication::add_to_pack(data, lambda_elast_pred_);
          Core::Communication::add_to_pack(data, lambda_plast_pred_);
          Core::Communication::add_to_pack(data, log_lambda_elast_pred_);
          Core::Communication::add_to_pack(data, log_lambda_plast_pred_);
          Core::Communication::add_to_pack(data, Qmat_elast_pred_);
          Core::Communication::add_to_pack(data, Qmat_plast_pred_);
          Core::Communication::add_to_pack(data, Qmat_plast_pred_rel_);
          Core::Communication::add_to_pack(data, Qvec_plast_pred_rel_);
          Core::Communication::add_to_pack(data, Rmat_elast_pred_);
          Core::Communication::add_to_pack(data, Rmat_plast_pred_);
          Core::Communication::add_to_pack(data, Rmat_plast_pred_rel_);
          Core::Communication::add_to_pack(data, Rvec_plast_pred_rel_);
          Core::Communication::add_to_pack(data, specific_defgrad_elast_pred_);
          Core::Communication::add_to_pack(data, specific_defgrad_plast_pred_);
        }

        //! unpack method
        void unpack(Core::Communication::UnpackBuffer& buffer)
        {
          Core::Communication::extract_from_pack(buffer, specific_defgrad_elast_pred_);
          Core::Communication::extract_from_pack(buffer, specific_defgrad_plast_pred_);
          Core::Communication::extract_from_pack(buffer, lambda_elast_pred_);
          Core::Communication::extract_from_pack(buffer, lambda_plast_pred_);
          Core::Communication::extract_from_pack(buffer, log_lambda_elast_pred_);
          Core::Communication::extract_from_pack(buffer, log_lambda_plast_pred_);
          Core::Communication::extract_from_pack(buffer, Qmat_elast_pred_);
          Core::Communication::extract_from_pack(buffer, Qmat_plast_pred_);
          Core::Communication::extract_from_pack(buffer, Qmat_plast_pred_rel_);
          Core::Communication::extract_from_pack(buffer, Qvec_plast_pred_rel_);
          Core::Communication::extract_from_pack(buffer, Rmat_elast_pred_);
          Core::Communication::extract_from_pack(buffer, Rmat_plast_pred_);
          Core::Communication::extract_from_pack(buffer, Rmat_plast_pred_rel_);
          Core::Communication::extract_from_pack(buffer, Rvec_plast_pred_rel_);
          Core::Communication::extract_from_pack(buffer, specific_defgrad_elast_pred_);
          Core::Communication::extract_from_pack(buffer, specific_defgrad_plast_pred_);
        }

        //! print method
        void print(std::ostream& os) const
        {
          std::cout << "PredictorDefgradDecomposition: \n";
          std::cout << "elastic - plastic: " << std::endl;
          std::cout << "lambda: [" << lambda_elast_pred_[0] << ", " << lambda_elast_pred_[1] << ", "
                    << lambda_elast_pred_[2] << "] - [" << lambda_plast_pred_[0] << ", "
                    << lambda_plast_pred_[1] << ", " << lambda_plast_pred_[2] << "]\n";
          std::cout << "Qvec_rel: [" << "0" << ", " << "0" << ", "
                    << "0" << "] - [" << Qvec_plast_pred_rel_(0) << ", " << Qvec_plast_pred_rel_(1)
                    << ", " << Qvec_plast_pred_rel_(2) << "]\n";
        }
      };

      //! enum class: deformation gradient decomposed and
      //! interpolated in the interpolation routine
      enum class DefgradType
      {
        elastic_defgrad,      ///< elastic deformation gradients are decomposed
        inv_plastic_defgrad,  ///< inverse plastic deformation gradients are decomposed
      };

      /**
       * @brief Set deformation gradient type used in the interpolation routine.
       *
       * @note We use either the inverse plastic or the elastic deformation
       * gradients depending on the employed rotation assignment types. E.g., if
       * the trial elastic rotation is maintained, then we decompose and interpolate the elastic
       * deformation gradient. Conversely, the plastic deformation gradients are decomposed when the
       * plastic rotation from the previous timestep is assumed to be preserved.
       *
       * @param[in] rot_assign_type Rotation assignment type for elastic and
       * plastic deformation gradients within the plastic predictor.
       *
       */
      DefgradType get_defgrad_type() { return defgrad_type_; }


      //! struct: specified point in interpolation space used to interpolate the
      //! inverse plastic deformation gradient (or the elastic deformation
      //! gradient, depending on the user specification)
      struct InterpolationPoint
      {
        double xi_lambda_1_;                          ///< interpolation factor for the first
                                                      /// <eigenvalue lambda_1
        double xi_lambda_2_;                          ///< interpolation factor for the second
                                                      ///< eigenvalue lambda_2
        std::array<double, 3> xi_rel_eigenvect_rot_;  ///< interpolation factor for the relative
                                                      ///< eigenvector rotation vector
        //! print method
        void print(std::ostream& os) const
        {
          os << "-- Interpolation point: lambda_1: " << xi_lambda_1_
             << "; lambda_2: " << xi_lambda_2_ << "; rel_eigenvect_rot: ["
             << xi_rel_eigenvect_rot_[0] << ", " << xi_rel_eigenvect_rot_[1] << ", "
             << xi_rel_eigenvect_rot_[2] << "] \n";
        }
      };

      //! enum class: action if an error is encountered in the plastic predictor update routine
      enum class PlasticPredUpdateErrorAction
      {
        shift_towards_elastic_pred,  ///< shift the current interpolation interval towards the
                                     ///< elastic predictor (e.g., in case of 0 plastic strain
                                     ///< increments)
        shift_towards_plastic_pred,  ///< shift the current interpolation interval towards the
                                     ///< plastic predictor (e.g., in case of overflow errors due to
                                     ///< high yield surface exceedances
        ambiguous,  ///< unclear what action should be taken; this may either be an uncaught bug in
                    ///< the state evaluation associated with the plastic predictor, or some
                    ///< scenario we have not yet thought about ...

      };

      static inline PlasticPredUpdateErrorAction get_plastic_pred_update_error_action(
          ErrorType error_type)
      {
        PlasticPredUpdateErrorAction out{PlasticPredUpdateErrorAction::ambiguous};
        switch (error_type)
        {
          case ErrorType::overflow_error:
            out = PlasticPredUpdateErrorAction::shift_towards_plastic_pred;
            break;
          case ErrorType::under_yield_surface:
            out = PlasticPredUpdateErrorAction::shift_towards_elastic_pred;
            break;

          case ErrorType::negative_plastic_strain:
          case ErrorType::no_flow_resistance:
          case ErrorType::no_plastic_incompressibility:
          case ErrorType::failed_solution_linear_system_lnl:
          case ErrorType::failed_determ_line_search_step:
          case ErrorType::no_convergence_local_newton:
          case ErrorType::singular_jacobian:
          case ErrorType::failed_solution_analytic_linearization:
          case ErrorType::failed_computation_flow_resistance:
          case ErrorType::failed_computation_flow_resistance_derivs:
          case ErrorType::failed_matrix_log_evaluation:
          case ErrorType::failed_matrix_exp_evaluation:
          case ErrorType::failed_right_cg_interpolation:
            out = PlasticPredUpdateErrorAction::ambiguous;
            break;

          default:
            FOUR_C_THROW("No action specified for error {} within the plastic predictor update!",
                EnumTools::enum_name(error_type));
        }
        FOUR_C_ASSERT_ALWAYS(out != PlasticPredUpdateErrorAction::ambiguous,
            "Action for error {} within the plastic predictor update is still ambiguous!",
            EnumTools::enum_name(error_type));

        return out;
      }

      //! interval scanning parameter set by the user \f$ k_{\mathrm{scan}} \f$
      const double k_scan_;

      //! current number of Local Newton Guess Interpolations
      unsigned int num_of_lngi_;

      //! maximum number of allowed
      //! Reinterpolations (including the initial Local Newton Guess Interpolation) before throwing
      //! error
      const unsigned int max_num_lngi_;

      //! minimum interpolation interval as a 2-norm \f$ \|  \mathbf{\xi}_{\text{upper}} -
      //! \mathbf{\xi}_{\text{upper}} \| \f$
      const double min_interp_interval_;

      // maximum allowed number number of Local Newton Guess Interpolation iterations
      static constexpr unsigned int MAX_NUM_PRED_ADAPT_ITERS = 100;

      //! current initial guess containing the inverse inelastic deformation
      //! gradient (components 0-8) and the plastic strain (component 9)
      Core::LinAlg::Matrix<10, 1> guess_inv_plast_defgrad_;

      /*!
       * @brief Constructor
       *
       * @param[in] k_scan Interval scanning parameter \f$ k_{\mathrm{scan}} \f$
       * @param[in] max_num_reinterp Maximum number of allowed
       * Reinterpolations (including the initial Local Newton Guess Interpolation) before throwing
       * error
       * @param[in] stretch_assign_type Stretch assignment type for the elastic
       * and plastic deformation gradient within the plastic predictor
       * @param[in] rot_assign_type Rotation assignment type for the elastic
       * and plastic deformation gradient within the plastic predictor
       *  @param[in] min_interp_interval Minimum interpolation interval upper -
       *  lower (2-norm in interpolation space)
       */
      LocalNewtonGuessInterpolation(const double k_scan, const unsigned int max_num_reinterp,
          const PlasticPredictorElasticStretchEigenvalType elastic_stretch_eigenval_type,
          const PlasticPredictorElasticStretchEigenvectRotType elastic_stretch_eigenvect_rot_type,
          const PlasticPredictorRotationType rot_type, const double min_interp_interval);

      //! setup method: set the correct number of Gauss Points to track the internal variables
      //! of the class
      void setup(const unsigned int num_gp);

      /*!
       * @brief Verify whether interpolation is still possible, based on the
       * set minimum interpolation interval, the set maximum number of interpolation
       * iterations, and the set maximum number of reinterpolations
       *
       */
      bool is_interpolation_possible(const unsigned int gp, const unsigned int num_interp_iters)
      {
        // check interpolation interval
        const double diff_bounds = get_diff_interp_points(
            get_lower_bound_interp_point(gp), get_upper_bound_interp_point(gp));
        bool check_min_interp_interval = (diff_bounds >= min_interp_interval_);
        if (!check_min_interp_interval)
        {
          std::cout << "INIT GUESS INTERPOLATION ERROR: difference between bounds: " << diff_bounds
                    << " < " << min_interp_interval_ << std::endl;
          return false;
        }

        // check number of reinterpolations
        bool check_num_reinterp = (num_of_lngi_ <= max_num_lngi_);
        if (!check_num_reinterp)
        {
          std::cout << "INIT GUESS INTERPOLATION ERROR: num of reinterpolations : " << num_of_lngi_
                    << " > " << max_num_lngi_ << std::endl;
          return false;
        }

        // check number of interpolation iterations
        bool check_interp_iters = (num_interp_iters <= MAX_NUM_PRED_ADAPT_ITERS);
        if (!check_interp_iters)
        {
          std::cout << "INIT GUESS INTERPOLATION ERROR: num of interpolation iters : "
                    << check_interp_iters << " > " << MAX_NUM_PRED_ADAPT_ITERS << std::endl;
          return false;
        }

        return true;
      }

      /*!
       * @brief Preevaluation method, performing reset tasks and setting the reference values
       * for interpolation
       *
       * @param[in] gp Gauss point index
       * @param[in] inv_plastic_defgrad_elast_pred inverse plastic deformation
       * gradient inside the elastic predictor
       * @param[in] inv_plastic_defgrad_plast_pred inverse plastic deformation
       * gradient inside the plastic predictor
       * @param[in] defgrad Current deformation gradient (current Local Newton
       * iteration of \f$ \left[ t_n, t_{n+1} \right] \f$)
       */
      void pre_evaluate(const unsigned int gp,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_elast_pred,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_plast_pred,
          const Core::LinAlg::Matrix<3, 3>& defgrad);

      /*!
       * @brief update method: update the internal variables based on the time step quantities of
       * the material.
       */
      void update();

      //! pack method
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack method
      void unpack(Core::Communication::UnpackBuffer& buffer);

      /*!
       * @brief Perform decompositions of inverse plastic | elastic deformation
       * gradient within the elastic and plastic predictors.
       *
       * @param[in] gp Gauss point index
       * @param[in] inv_plastic_defgrad_elast_pred inverse plastic deformation
       * gradient within the elastic predictor
       * @param[in] inv_plastic_defgrad_plast_pred inverse plastic deformation
       * gradient within the plastic predictor
       * @param[in] defgrad Current deformation gradient (current Local Newton
       * iteration of \f$ \left[ t_n, t_{n+1} \right] \f$)
       *
       */
      void perform_predictor_decomposition(const unsigned int gp,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_elast_pred,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_plast_pred,
          const Core::LinAlg::Matrix<3, 3>& defgrad);

      /*!
       * @brief Retrieves the current interpolation point at a specified Gauss point, based on the
       * current xi values.
       *
       * @param[in] gp Gauss point index
       * @return Current interpolation point
       */
      InterpolationPoint get_curr_interp_point(const unsigned int gp) const
      {
        return InterpolationPoint{.xi_lambda_1_ = all_component_interp_lambda_1_[gp].current_xi_[0],
            .xi_lambda_2_ = all_component_interp_lambda_2_[gp].current_xi_[0],
            .xi_rel_eigenvect_rot_ = all_component_interp_rel_eigenvect_rot_[gp].current_xi_};
      };

      /*!
       *
       * @brief Retrieves the previous timestep interpolation point at a specified Gauss point,
       * based on the last xi values.
       *
       * @param[in] gp Gauss point index
       * @return Current interpolation point
       */
      InterpolationPoint get_last_interp_point(const unsigned int gp) const
      {
        return InterpolationPoint{.xi_lambda_1_ = all_component_interp_lambda_1_[gp].last_xi_[0],
            .xi_lambda_2_ = all_component_interp_lambda_2_[gp].last_xi_[0],
            .xi_rel_eigenvect_rot_ = all_component_interp_rel_eigenvect_rot_[gp].last_xi_};
      };

      /*!
       *
       * @brief Retrieves the optimal interpolation point at a specified Gauss point associated with
       * the previous timestep
       *
       * @param[in] gp Gauss point index
       * @return Current interpolation point
       */
      InterpolationPoint get_optimal_interp_point(const unsigned int gp) const
      {
        return InterpolationPoint{.xi_lambda_1_ = all_component_interp_lambda_1_[gp].optimal_xi_[0],
            .xi_lambda_2_ = all_component_interp_lambda_2_[gp].optimal_xi_[0],
            .xi_rel_eigenvect_rot_ = all_component_interp_rel_eigenvect_rot_[gp].optimal_xi_};
      };

      /*!
       * @brief Retrieves the interpolation point associated with the lower interpolation interval
       * bound at a specified Gauss point
       *
       * @param[in] gp Gauss point index
       * @return Interpolation point of the lower bound
       */
      InterpolationPoint get_lower_bound_interp_point(const unsigned int gp) const
      {
        return InterpolationPoint{.xi_lambda_1_ = all_component_interp_lambda_1_[gp].xi_l_[0],
            .xi_lambda_2_ = all_component_interp_lambda_2_[gp].xi_l_[0],
            .xi_rel_eigenvect_rot_ = all_component_interp_rel_eigenvect_rot_[gp].xi_l_};
      };


      /*!
       * @brief Retrieves the interpolation point associated with the upper interpolation interval
       * bound at a specified Gauss point
       *
       * @param[in] gp Gauss point index
       * @return Interpolation point of the upper bound
       */
      InterpolationPoint get_upper_bound_interp_point(const unsigned int gp) const
      {
        return InterpolationPoint{.xi_lambda_1_ = all_component_interp_lambda_1_[gp].xi_u_[0],
            .xi_lambda_2_ = all_component_interp_lambda_2_[gp].xi_u_[0],
            .xi_rel_eigenvect_rot_ = all_component_interp_rel_eigenvect_rot_[gp].xi_u_};
      };

      /*!
       * @brief Sets the current interpolation point at specified Gauss point.
       *
       * @param[in] gp Gauss point index
       * @param[in] interp_point Interpolation point to be specified as current
       */
      void set_curr_interp_point(const unsigned int gp, const InterpolationPoint& interp_point)
      {
        all_component_interp_lambda_1_[gp].current_xi_ = {interp_point.xi_lambda_1_};
        all_component_interp_lambda_2_[gp].current_xi_ = {interp_point.xi_lambda_2_};
        all_component_interp_rel_eigenvect_rot_[gp].current_xi_ =
            interp_point.xi_rel_eigenvect_rot_;
      }

      /*!
       * @brief Sets the last interpolation point, associated with the previous timestep, at
       * specified Gauss point.
       *
       * @param[in] gp Gauss point index
       * @param[in] interp_point Interpolation point to be specified as last
       */
      void set_last_interp_point(const unsigned int gp, const InterpolationPoint& interp_point)
      {
        all_component_interp_lambda_1_[gp].last_xi_ = {interp_point.xi_lambda_1_};
        all_component_interp_lambda_2_[gp].last_xi_ = {interp_point.xi_lambda_2_};
        all_component_interp_rel_eigenvect_rot_[gp].last_xi_ = interp_point.xi_rel_eigenvect_rot_;
      }

      /*!
       * @brief Sets the optimal interpolation point, associated with the previous timestep, at
       * specified Gauss point.
       *
       * @param[in] gp Gauss point index
       * @param[in] interp_point Interpolation point to be specified as optimal
       */
      void set_optimal_interp_point(const unsigned int gp, const InterpolationPoint& interp_point)
      {
        all_component_interp_lambda_1_[gp].optimal_xi_ = {interp_point.xi_lambda_1_};
        all_component_interp_lambda_2_[gp].optimal_xi_ = {interp_point.xi_lambda_2_};
        all_component_interp_rel_eigenvect_rot_[gp].optimal_xi_ =
            interp_point.xi_rel_eigenvect_rot_;
      }


      /*!
       * @brief Sets the interpolation point associated with the lower
       * interpolation interval bound at specified Gauss point.
       *
       * @param[in] gp Gauss point index
       * @param[in] interp_point Interpolation point to be specified as lower
       * bound
       */
      void set_lower_bound_interp_point(
          const unsigned int gp, const InterpolationPoint& interp_point)
      {
        all_component_interp_lambda_1_[gp].xi_l_ = {interp_point.xi_lambda_1_};
        all_component_interp_lambda_2_[gp].xi_l_ = {interp_point.xi_lambda_2_};
        all_component_interp_rel_eigenvect_rot_[gp].xi_l_ = interp_point.xi_rel_eigenvect_rot_;
      }

      /*!
       * @brief Sets the interpolation point associated with the upper
       * interpolation interval bound at specified Gauss point.
       *
       * @param[in] gp Gauss point index
       * @param[in] interp_point Interpolation point to be specified as upper
       * bound
       */
      void set_upper_bound_interp_point(
          const unsigned int gp, const InterpolationPoint& interp_point)
      {
        all_component_interp_lambda_1_[gp].xi_u_ = {interp_point.xi_lambda_1_};
        all_component_interp_lambda_2_[gp].xi_u_ = {interp_point.xi_lambda_2_};
        all_component_interp_rel_eigenvect_rot_[gp].xi_u_ = interp_point.xi_rel_eigenvect_rot_;
      }

      /*!
       * @brief Retrieves the difference (2-norm) between two interpolation points.
       *
       * @param[in] interp_point_1 First interpolation point
       * @param[in] interp_point_2 Second interpolation point
       */
      static double get_diff_interp_points(
          const LocalNewtonGuessInterpolation::InterpolationPoint& interp_point_1,
          const LocalNewtonGuessInterpolation::InterpolationPoint& interp_point_2)
      {
        const double diff_lambda_1 = (interp_point_2.xi_lambda_1_ - interp_point_1.xi_lambda_1_) *
                                     (interp_point_2.xi_lambda_1_ - interp_point_1.xi_lambda_1_);
        const double diff_lambda_2 = (interp_point_2.xi_lambda_2_ - interp_point_1.xi_lambda_2_) *
                                     (interp_point_2.xi_lambda_2_ - interp_point_1.xi_lambda_2_);
        const double diff_rot_1 =
            (interp_point_2.xi_rel_eigenvect_rot_[0] - interp_point_1.xi_rel_eigenvect_rot_[0]) *
            (interp_point_2.xi_rel_eigenvect_rot_[0] - interp_point_1.xi_rel_eigenvect_rot_[0]);
        const double diff_rot_2 =
            (interp_point_2.xi_rel_eigenvect_rot_[1] - interp_point_1.xi_rel_eigenvect_rot_[1]) *
            (interp_point_2.xi_rel_eigenvect_rot_[1] - interp_point_1.xi_rel_eigenvect_rot_[1]);
        const double diff_rot_3 =
            (interp_point_2.xi_rel_eigenvect_rot_[2] - interp_point_1.xi_rel_eigenvect_rot_[2]) *
            (interp_point_2.xi_rel_eigenvect_rot_[2] - interp_point_1.xi_rel_eigenvect_rot_[2]);

        return std::sqrt(diff_lambda_1 + diff_lambda_2 + diff_rot_1 + diff_rot_2 + diff_rot_3);
      }

      /*!
       * @brief Add interpolation points.
       *
       */
      static LocalNewtonGuessInterpolation::InterpolationPoint add_interpolation_points(
          const double scalar_a,
          const LocalNewtonGuessInterpolation::InterpolationPoint& interp_point_a,
          const double scalar_b,
          const LocalNewtonGuessInterpolation::InterpolationPoint& interp_point_b)
      {
        return LocalNewtonGuessInterpolation::InterpolationPoint{
            .xi_lambda_1_ =
                scalar_a * interp_point_a.xi_lambda_1_ + scalar_b * interp_point_b.xi_lambda_1_,
            .xi_lambda_2_ =
                scalar_a * interp_point_a.xi_lambda_2_ + scalar_b * interp_point_b.xi_lambda_2_,
            .xi_rel_eigenvect_rot_ = {
                scalar_a * interp_point_a.xi_rel_eigenvect_rot_[0] +
                    scalar_b * interp_point_b.xi_rel_eigenvect_rot_[0],
                scalar_a * interp_point_a.xi_rel_eigenvect_rot_[1] +
                    scalar_b * interp_point_b.xi_rel_eigenvect_rot_[1],
                scalar_a * interp_point_a.xi_rel_eigenvect_rot_[2] +
                    scalar_b * interp_point_b.xi_rel_eigenvect_rot_[2],
            }};
      }


      /*!
       * @brief Interpolate inverse plastic deformation gradient between the
       * elastic and the plastic predictor, given the
       * current several interpolation factors.
       *
       * @note 1. Depending on the user-specified rotation assignment, it is
       * possible that the elastic deformation gradient is actually
       * interpolated, and the inverse plastic deformation gradient is simply
       * computed based on it.
       * 2. The eigenvalues are interpolated using the logarithmic weighted average
       * (see Satheesh et al. 2022, 10.1002/nme.7373) with linear weighting
       * between the predictors.
       *
       * @param[in] gp Gauss point index
       * @param[in] defgrad Current deformation gradient (current Local Newton
       * iteration of \f$ \left[ t_n, t_{n+1} \right] \f$)
       * @param[in] interp_point Interpolation point to be used.
       * @param[in] inv_defgrad Inverse of the current deformation gradient (current Local Newton
       * iteration of \f$ \left[ t_n, t_{n+1} \right] \f$)
       *
       */
      Core::LinAlg::Matrix<3, 3> interpolate_inv_plastic_defgrad(const unsigned int gp,
          const Core::LinAlg::Matrix<3, 3>& defgrad, const InterpolationPoint& interp_point,
          const Core::LinAlg::Matrix<3, 3>& inv_defgrad);

      /*!
       * @brief Adapt interpolation parameters  and interpolation intervals
       * based on evaluation error
       *
       * @param[in] gp Gauss point index
       * @param[in] eval_err_type type of evaluation error necessitating
       * an adaptation of the parameter and the interval
       *
       */
      void adapt_interpolation_intervals(const unsigned int gp, const ErrorType eval_err_type);

      /*!
       * @brief Adapt interpolation parameters based on the current
       * parameter bounds
       *
       * @param[in] gp Gauss point index
       *
       */
      void adapt_interpolation_parameters(const unsigned int gp);

      /**
       * @brief Compute optimal interpolation factors based on the solution of
       * the Local Newton Loop.
       *
       * @param[in] gp Gauss point index
       * @param[in] inv_plastic_defgrad_solution Solution of the Local Newton
       * loop: inverse plastic deformation gradient
       * @param[in] defgrad Deformation gradient at time \f$ t_{n+1} \f$
       * serving as input for Local Newton loop
       *
       */
      InterpolationPoint compute_optimal_interp_factors(
          const PredictorDefgradDecomposition& predictor_defgrad_decomposition, unsigned int gp,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_solution,
          const Core::LinAlg::Matrix<3, 3>& defgrad);

      [[nodiscard]] const std::vector<PredictorDefgradDecomposition>&
      get_curr_pred_decomp_specific_defgrad() const
      {
        return curr_pred_decomp_specific_defgrad_;
      }

      [[nodiscard]] const std::vector<PredictorDefgradDecomposition>&
      get_last_pred_decomp_specific_defgrad() const
      {
        return last_pred_decomp_specific_defgrad_;
      }


     private:
      //! type of elastic stretch eigenvalues within plastic predictor
      PlasticPredictorElasticStretchEigenvalType plast_pred_elast_stretch_eigenval_type_;

      //! type of elastic stretch eigenvector rotation within plastic predictor
      PlasticPredictorElasticStretchEigenvectRotType plast_pred_elast_stretch_eigenvect_rot_type_;

      //! type of rotation within plastic predictor
      PlasticPredictorRotationType plast_pred_rot_type_;

      //! decomposed deformation gradient type
      const DefgradType defgrad_type_;

      //! current timestep predictor decompositions for each GP of deformation gradient (with
      //! specified type: elastic, inverse plastic, ...) within elastic and plastic predictors
      //!--> MOVE TO PRIVATE AFTER REMOVING CONSISTENCY CHECKS
      std::vector<PredictorDefgradDecomposition> curr_pred_decomp_specific_defgrad_;

      //! last timestep predictor decompositions for each GP of deformation gradient (with
      //! specified type: elastic, inverse plastic, ...) within elastic and plastic predictors
      //!--> MOVE TO PRIVATE AFTER REMOVING CONSISTENCY CHECKS
      std::vector<PredictorDefgradDecomposition> last_pred_decomp_specific_defgrad_;



      /**
       * @brief Component interpolator storing arrays of interpolation parameters
       *        and bounds for a single interpolation component.
       *
       * @tparam num_items Length of each interpolation parameter array.
       */
      template <std::size_t num_items>
      struct ComponentInterpolator
      {
        //! interpolation factor for the current timestep
        std::array<double, num_items> current_xi_;

        //! interpolation factor from previous timestep
        std::array<double, num_items> last_xi_;

        //! optimal interpolation factor based on the solution from the previous
        //! timstep
        std::array<double, num_items> optimal_xi_;

        //! lower bound for the current interpolation parameter
        std::array<double, num_items> xi_l_;

        //! upper bound for the current interpolation parameter
        std::array<double, num_items> xi_u_;
      };

      //! interpolator for each GP for first eigenvalue \f$ \lambda_1 \f$ of either the
      //! elastic or the plastic defgrad (depending on rotation assignment)
      std::vector<ComponentInterpolator<1>> all_component_interp_lambda_1_;

      //! interpolator for each GP for second eigenvalue \f$ \lambda_2 \f$ of either the
      //! elastic or the plastic defgrad (depending on rotation assignment)
      std::vector<ComponentInterpolator<1>> all_component_interp_lambda_2_;

      //! interpolator for each GP for relative eigenvector rotation vector \f$
      //! \boldsymbol{q}_{\mathrm{rel}} \f$ of either the elastic or the plastic defgrad
      //! (depending on rotation assignment)
      std::vector<ComponentInterpolator<3>> all_component_interp_rel_eigenvect_rot_;

      //! control variable: should eigenvector rotations be interpolated (if
      //! not, take the eigenvector rotation of the elastic predictor)
      const bool interpolate_elastic_stretch_eigenvect_rot_;

      //! control variable: should rotations be interpolated (if
      //! not, take the rotation of the elastic predictor)
      const bool interpolate_rot_;
    };

    //! struct with local substepping utilities
    struct LocalSubsteppingUtils
    {
      //! current time parameter ranging from 0 to the problem time step \f$ \Delta t \f$
      double t_;
      //! counter of evaluated substeps
      unsigned int substep_counter_;
      //! current substep size
      double curr_dt_;
      //! number of times the problem time step \f$ \Delta t \f$ has been halved
      unsigned int time_step_halving_counter_;
      //!  current total number of substeps to be evaluated within the time step \f$ \Delta t
      //! \f$; this is not always given by time_step_halving_counter, since the
      //! halving does not have to be uniform (e.g. we could halve the time step twice and still
      //! have 3 substeps to evaluate instead of 4, i.e. if the first substep was evaluable
      //! numerically, but the second substep not, leading to another halving of the substep
      //! length)
      unsigned int total_num_of_substeps_;

      //! reset routine: basically, create a new empty object
      void reset();
    };

    /// class containing utilities for general analysis of the material
    /// time integration (including Local Newton loop, Local Newton
    /// Guess Interpolation, line search):
    /// error types, number of line searches, timers, ... Currently only
    /// employed for single-element single-processor simulations.
    class GeneralLocalTimIntAnalysisUtils
    {
     public:
      //! struct: numbers of iterations, steps, ... for the different
      //! computation methods used (such as e.g., LNGI)
      struct NumItersAndSteps
      {
        //! number of LNL steps for the current timestep evaluation (LNL)
        unsigned int eval_num_of_LNL_steps_ = 0;

        //! total number of LNL steps over all time steps
        unsigned int total_num_of_LNL_steps_ = 0;

        //! number of iterations for the current timestep evaluation (LNL)
        unsigned int eval_num_of_iters_ = 0;

        //! total number of LNL iterations over all time steps
        unsigned int total_num_of_iters_ = 0;

        //! number of reinterpolations for the current timestep evaluation (LNL)
        unsigned int eval_num_of_lngi_reinterp_ = 0;

        //! total number of Local Newton Guess Reinterpolations over all time steps
        unsigned int total_num_of_lngi_reinterp_ = 0;

        //! number of iterations spent in the Local Newton Guess Interpolations for the
        //! current timestep evaluation (including Reinterpolations)
        unsigned int eval_num_of_lngi_iters_ = 0;

        //! total number of iterations spent in the Local Newton Guess
        //! Interpolation
        //! over all time steps (including Reinterpolation)
        unsigned int total_num_of_lngi_iters_ = 0;

        //! number of iterations spent in the Local Newton Guess Interpolation for the
        //! current timestep evaluation (LNL), in the specific case of Reinterpolation
        unsigned int eval_num_of_reinterp_iters_ = 0;

        //! total number of iterations spent in the Local Newton Guess Interpolation
        //! over all time steps,
        //! in the specific case of Reinterpolation
        unsigned int total_num_of_reinterp_iters_ = 0;

        //! number of line searches for the current timestep evaluation (LNL)
        unsigned int eval_num_of_line_search_ = 0;

        //! line search: the number of times the step size \f$ \alpha \f$ of
        //! the last iteration (Local Newton Loop) deviates from 1.0
        //! (currently evaluated time step)
        unsigned int eval_num_of_alpha_neq_1_last_iter_ = 0;

        //! line search: the number of times the step size \f$ \alpha \f$
        //! deviates from 1.0 in all iterations of the Local Newton Loop
        //! (currently evaluated time step)
        unsigned int eval_num_of_alpha_neq_1_ = 0;

        //! total number of LNL line searches over all time steps
        unsigned int total_num_of_line_search_ = 0;

        //! line search: total number of times the step size \f$ \alpha \f$ of
        //! the last iteration (Local Newton Loop) deviates from 1.0
        unsigned int total_num_of_alpha_neq_1_last_iter_ = 0;

        //! line search: the number of times the step size \f$ \alpha \f$
        //! deviates from 1.0 in all iterations of the Local Newton Loop
        //! (currently evaluated time step)
        unsigned int total_num_of_alpha_neq_1_ = 0;

        //! number of iterations of the line searches for the current timestep evaluation (LNL)
        unsigned int eval_num_of_line_search_iters_ = 0;

        //! total number of line search iterations over all time steps
        unsigned int total_num_of_line_search_iters_ = 0;
      };
      NumItersAndSteps num_iters_and_steps_;

      //! struct: interpolation factors for the LNGI
      struct LNGIFactors
      {
        //! Local Newton Guess Interpolation factor for eigenvalue \f$ \lambda_1 \f$ obtained from
        //! the Local Newton Guess Interpolation routine (for set GP, current time step, last global
        //! iteration)
        double curr_lngi_factor_lambda_1_ = 0;

        //! Local Newton Guess Interpolation factor for eigenvalue \f$ \lambda_2 \f$ obtained from
        //! the Local Newton Guess Interpolation routine (for set GP, current time step, last global
        //! iteration)
        double curr_lngi_factor_lambda_2_ = 0;

        //! Local Newton Guess Interpolation factor for rotation vector (component
        //! 0) associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$ obtained
        //! from the Local Newton Guess Interpolation routine (for set GP, current time step, last
        //! global iteration)
        double curr_lngi_factor_eigenvect_rot_comp_0_ = 0;

        //! Local Newton Guess Interpolation factor for rotation vector (component
        //! 1) associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$ obtained
        //! from the Local Newton Guess Interpolation routine (for set GP, current time step, last
        //! global iteration)
        double curr_lngi_factor_eigenvect_rot_comp_1_ = 0;

        //! Local Newton Guess Interpolation factor for rotation vector (component
        //! 2) associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$ obtained
        //! from the Local Newton Guess Interpolation routine (for set GP, current time step, last
        //! global iteration)
        double curr_lngi_factor_eigenvect_rot_comp_2_ = 0;

        //! Local Newton Guess Interpolation factor (eigenvalue \f$ \lambda_1 \f$) obtained from the
        //! Local Newton Guess Interpolation routine (for set GP, current time step, maximum over
        //! all global iterations)
        double curr_max_lngi_factor_lambda_1_ = 0;

        //! Local Newton Guess Interpolation factor (eigenvalue \f$ \lambda_2 \f$) obtained from the
        //! Local Newton Guess Interpolation routine (for set GP, current time step, maximum over
        //! all global iterations)
        double curr_max_lngi_factor_lambda_2_ = 0;

        //! Local Newton Guess Interpolation factor (rotation vector associated
        //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
        //! component 0) obtained for set GP, current
        //! time step, maximum over all global iterations
        double curr_max_lngi_factor_eigenvect_rot_comp_0_ = 0;

        //! Local Newton Guess Interpolation factor (rotation vector associated
        //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
        //! component 1) obtained for set GP, current
        //! time step, maximum over all global iterations
        double curr_max_lngi_factor_eigenvect_rot_comp_1_ = 0;

        //! Local Newton Guess Interpolation factor (rotation vector associated
        //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
        //! component 2) obtained for set GP, current
        //! time step, maximum over all global iterations
        double curr_max_lngi_factor_eigenvect_rot_comp_2_ = 0;

        //! optimal Local Newton Guess Interpolation factor (eigenvalue \f$ \lambda_1 \f$) obtained
        //! from the time step solution (for GP 0 of element 0 after the current time step)
        double optimal_lngi_factor_lambda_1_ = 0;

        //! optimal Local Newton Guess Interpolation factor (eigenvalue \f$ \lambda_2 \f$) obtained
        //! from the time step solution (for GP 0 of element 0 after the current time step)
        double optimal_lngi_factor_lambda_2_ = 0;

        //! optimal Local Newton Guess Interpolation factor (rotation vector associated
        //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
        //! component 0) obtained from the
        //! time step solution (for GP 0 of element 0 after the current time step)
        double optimal_lngi_factor_eigenvect_rot_comp_0_ = 0;

        //! optimal Local Newton Guess Interpolation factor (rotation vector associated
        //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
        //! component 1) obtained from the
        //! time step solution (for GP 0 of element 0 after the current time step)
        double optimal_lngi_factor_eigenvect_rot_comp_1_ = 0;

        //! optimal Local Newton Guess Interpolation factor (rotation vector associated
        //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
        //! component 2) obtained from the
        //! time step solution (for GP 0 of element 0 after the current time step)
        double optimal_lngi_factor_eigenvect_rot_comp_2_ = 0;

        //! Local Newton residual obtained from the optimal Local Newton Guess Interpolation factor
        double lnl_res_optimal_lngi_factor_ = 0;
      };
      LNGIFactors lngi_factors_;

      //! struct: contains timers for the different
      //! computations performed
      struct Timers
      {
        //! timer for the return mapping in the current timestep
        Teuchos::Time eval_teuchos_timer_rma_{
            "InelasticDefgradTransvIsotropElastViscoplast::Return Mapping"};

        //! boolean: should the timer for return mapping be used?
        const bool use_teuchos_timer_rma_ = false;

        //! timer for preparing + updating the Local Newton Guess Interpolation for current / next
        //! timestep
        Teuchos::Time eval_teuchos_timer_lngi_preparation_{
            "InelasticDefgradTransvIsotropElastViscoplast::Local Newton Guess Interpolation (only "
            "initial interpolation, no reinterpolation)"};

        //! boolean: should the timer for preparing + updating for the Local Newton Guess
        //! Interpolation be used?
        const bool use_teuchos_timer_lngi_preparation_ = false;
      };
      Timers timers_;

      //! struct: contains time measurements using the timers for the different
      //! computations performed
      struct TimeMeasurements
      {
        //! evaluation time spent in the return mapping (current time step)
        double eval_time_rma_;

        //! total time spent in the return mapping over all time steps
        double total_time_rma_;

        //! evaluation time spent for preparing + updating the Local Newton Guess
        //! Interpolation for the current / next timestep
        double eval_time_lngi_preparation_;

        //! total time spent for preparing the next timestep for the Local Newton Guess
        //! Interpolation (over all timesteps)
        double total_time_lngi_preparation_;
      };
      TimeMeasurements time_measurements_;

      //! error map (how many times an error occurs) of the current timestep evaluation, from the
      //! first preevaluate of this time step to the first preevaluate of the next
      std::map<ErrorType, unsigned int> eval_error_map_ = {
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

      //! error map (how many times an error occurs) of the total evaluation
      std::map<ErrorType, unsigned int> total_error_map_ = {
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
          {ErrorType::failed_right_cg_interpolation, 0},
          {ErrorType::under_yield_surface, 0},
      };

      //! simulation time instant
      double sim_time_ = 0.0;
      //! simulation time step index
      int sim_timestep_ = 0;

      //! reset called for current time step?
      bool reset_called_ = false;

      //! how often was the update method called? (maximum:
      //! num_of_global_elements, if only one processor
      //! is considered)
      int num_update_calls_ = 0;

      //! runtime csv writer
      std::optional<Core::IO::RuntimeCsvWriter> csv_writer_;

      //! boolean: control variable to check whether number of iters,
      //! steps, errors, ... should be incremented (to be used in benchmarking, so that
      //! repeating a procedure should not reincrement iters and steps)
      bool increment_vars_ = true;

      //! initialize csv writer
      void init_csv_writer();

      //! reset method: reset the stored internal variables for a new
      //! evaluation / new timestep
      void reset();

      //! update total values based on the evaluated values
      void update_total();

      //! write the stored internal variables to csv
      void write_to_csv();

      //! output routine of the csv writer in the case of an error
      //! during the Local Newton Loop routine
      void output_error_local_newton_loop(unsigned int step_counter);
    };

    //! struct containing quantities computed from a given elasticity/plasticity state;
    //! given: current right Cauchy-Green deformation tensor, inelastic deformation gradient and
    //! plastic strain at the previous time instant
    struct StateQuantities
    {
      // ----- current state quantities (for the evaluated Gauss points) ----- //

      //! elastic right Cauchy-Green deformation tensor
      Core::LinAlg::Matrix<3, 3> curr_CeM_{Core::LinAlg::Initialization::zero};

      //! isotropic stress factors
      Core::LinAlg::Matrix<3, 1> curr_gamma_{Core::LinAlg::Initialization::zero};

      //! isotropic constitutive tensor factors
      Core::LinAlg::Matrix<8, 1> curr_delta_{Core::LinAlg::Initialization::zero};

      //! elastic 2nd PK stress tensors (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<3, 3> curr_SeM_{Core::LinAlg::Initialization::zero};

      //! elastic stiffness tensor (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<6, 6> curr_dSedCe_{Core::LinAlg::Initialization::zero};

      //! deviatoric, symmetric part of the Mandel stress tensor
      Core::LinAlg::Matrix<3, 3> curr_Me_dev_sym_M_{Core::LinAlg::Initialization::zero};

      //! equivalent tensile stress
      double curr_equiv_stress_{0.0};

      //! equivalent plastic strain rate
      double curr_equiv_plastic_strain_rate_{0.0};

      //! plastic flow direction tensor
      Core::LinAlg::Matrix<3, 3> curr_NpM_{Core::LinAlg::Initialization::zero};

      //! plastic stretching tensor
      Core::LinAlg::Matrix<3, 3> curr_dpM_{Core::LinAlg::Initialization::zero};

      //! plastic velocity gradient tensor
      Core::LinAlg::Matrix<3, 3> curr_lpM_{Core::LinAlg::Initialization::zero};

      //! plastic update tensor
      Core::LinAlg::Matrix<3, 3> curr_EpM_{Core::LinAlg::Initialization::zero};
    };


    //! struct containing specific derivatives of quantities computed from a given
    //! elasticity/plasticity state; given: current right Cauchy-Green deformation tensor,
    //! inelastic deformation gradient and plastic strain at the previous time instant
    struct StateQuantityDerivatives
    {
      // ----- current state variable derivatives (for the evaluated Gauss points)----- //

      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the inverse
      //! inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dCediFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the right
      //! Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dCedC_{Core::LinAlg::Initialization::zero};

      //! derivatives of the equivalent tensile stress w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<1, 9> curr_dequiv_stress_diFin_{Core::LinAlg::Initialization::zero};
      //! derivatives of the equivalent tensile stress w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress form)
      Core::LinAlg::Matrix<1, 6> curr_dequiv_stress_dC_{Core::LinAlg::Initialization::zero};

      //! derivative of the deviatoric, symmetric part of the Mandel stress tensor w.r.t. the
      //! inverse inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dMe_dev_sym_diFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the deviatoric, symmetric part of the Mandel stress tensor w.r.t. the
      //! right Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dMe_dev_sym_dC_{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic strain rate w.r.t. the equivalent stress
      double curr_dpsr_dequiv_stress_{0.0};
      //! derivative of the plastic strain rate w.r.t. the equivalent plastic strain
      double curr_dpsr_depsp_{0.0};

      //! derivative of the plastic stretching tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_ddpdiFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the equivalent plastic strain (Voigt
      //! stress form)
      Core::LinAlg::Matrix<6, 1> curr_ddpdepsp_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_ddpdC_{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic velocity gradient tensor w.r.t. the inverse inelastic
      //! deformation gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dlpdiFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the equivalent plastic strain
      //! (Voigt notation)
      Core::LinAlg::Matrix<9, 1> curr_dlpdepsp_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the right Cauchy-Green
      //! deformation tensor (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dlpdC_{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic update tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dEpdiFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the equivalent plastic strain (Voigt
      //! notation)
      Core::LinAlg::Matrix<9, 1> curr_dEpdepsp_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the right Cauchy-Green deformation tensor
      //! (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dEpdC_{Core::LinAlg::Initialization::zero};
    };



    /// enum class for state quantity evaluations in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// plastic strain rate,...)
    enum class StateQuantityEvalType
    {
      FullEval,  ///< full evaluation (full call of the evaluate_state_quantities method)
      PlasticStrainRateOnly,  ///< return in evaluate_state_quantities once the plastic strain
                              ///< rate has been evaluated
      EquivStressOnly,        ///< return in evaluate_state_quantities once the
                              ///< equivalent stress has been evaluated
    };

    /// enum class for evaluations of the state quantity derivatives in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// derivatives of the plastic strain rate,...)
    enum class StateQuantityDerivEvalType
    {
      FullEval,  ///< full evaluation (full call of the evaluate_state_quantity_derivatives
                 ///< method)
      PlasticStrainRateDerivsOnly,  ///< return in evaluate_state_quantity_derivatives once the
                                    ///< derivatives of the plastic strain rate have been
                                    ///< evaluated
      EquivStressDerivsOnly,  ///< return in evaluate_state_quantities once the derivatives of the
                              ///< equivalent stress has been evaluated
    };


    // enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonConvCheck
    {
      ResidualOnly,   ///< only verify convergence based on the absolute value of the Local Newton
                      ///< residual
      IncrementOnly,  ///< only verify convergence based on the 2-norm of the Local Newton
                      ///< solution increment
      ResidualAndIncrement,  ///< verify convergence based on both the Local Newton residual and
                             ///< the solution increment
    };



    // enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonDiverCont
    {
      Stop,      ///< stop the simulation entirely
      Continue,  ///<  continue the simulation, and display warning in regards to the current
                 ///<  state
                 ///< within the Local Newton Loop
      ContinueWithSafeGuard  ///< continue the simulation only if the convergence tolerances are
                             ///< not exceeded excessively
    };


    //! struct containing settings and iteration data from the Local Newton-Raphson
    //! Loop (time integration of the viscoplasticity equations)
    //! used for Gauss-Point output. In contrast to
    //! GeneralLocalTimIntAnalysisUtils which tracks general information
    //! over time steps (such as how many iterations were performed for
    //! a specific time step), this utility struct considers the data
    //! for each specific iteration of the Local Newton-Raphson Loop.
    struct LocalNewtonData
    {
      /*!
       *   @brief Constructor
       *
       * @param[in] res_tol tolerance for the Local Newton-Raphson
       * scheme (absolute residual value)
       * @param[in] incr_tol tolerance for the Local Newton-Raphson
       * scheme (2-norm of solution increment)
       * @param[in] conv_check convergence check strategy for the Local
       * Newton Loop
       * @param[in] diver_cont strategy for dealing with divergence of the Local
       * Newton Loop
       *
       */
      //! constructor of Local Newton data, based on the
      LocalNewtonData(const double res_tol, const double incr_tol,
          const LocalNewtonConvCheck conv_check, const LocalNewtonDiverCont diver_cont);

      //! convergence tolerance of the Local Newton Loop (absolute
      //! residual value)
      const double res_tol_;

      //! convergence tolerance of the Local Newton Loop (2-norm of the
      //! solution increment)
      const double incr_tol_;

      //! convergence check strategy of the Local Newton Loop
      const LocalNewtonConvCheck conv_check_;

      //! strategy for dealing with divergence of the Local Newton Loop
      const LocalNewtonDiverCont diver_cont_;

      //! maximum number of Local Newton Loop iterations
      static constexpr unsigned max_iter_ = 100;

      //! maximum exceedance factor of the residual tolerance (to be used when
      //! using the divergence management strategy for continuation with
      //! safeguard)
      static constexpr double max_exceedance_fact_res_tol_ = 1.0e3;

      //! maximum exceedance of the solution increment tolerance (to be used when
      //! using the divergence management strategy for continuation with
      //! safeguard)
      static constexpr double max_exceedance_fact_incr_tol_ = 1.0e2;

      //! current LNL iteration index
      unsigned int iter_;

      //! do we have Gauss point output every global iteration?
      bool is_Gauss_point_output_every_global_iter_ = false;

      //! number of LNL iterations for the current timestep; vector of GP values
      std::vector<unsigned int> num_iter_curr_timestep_;

      //! resize all relevant vectors based on the number of Gauss
      //! points known only after setting up the problem -> each vector
      //! item gets the same value for now
      void set_num_of_gp(const unsigned int num_of_gp);

      //! reset all arrays holding values for all iterations (for a
      //! given Gauss point)
      void reset_all_iteration_data(const unsigned int gp);


      // maybe we need some pack and unpack methods perspectively? If
      // this is to be used consistently in the future...-> would mainly
      // concern the global iteration / time step tracker, but nothing else.
    };

    // display / log evaluation warnings
#define DISPLAY_WARNINGS ;
  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
