// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace
{
  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;
  namespace LocalNewtonLineSearch = Core::Utils::LineSearch;

  [[nodiscard]] double merit_from_residual(const Core::LinAlg::Matrix<10, 1>& residual)
  {
    return 0.5 * residual.dot(residual);
  }

  // LocalNewtonLineSearchParams' per-algorithm fields (armijo, goldstein, wolfe, ...) are already
  // typed as the generic LineSearch structs directly (see *_service.hpp), so only alpha_init and
  // max_iter -- which stay flat scalars on LocalNewtonLineSearchParams -- need bundling into the
  // shape LineSearch's constructors expect.
  [[nodiscard]] LocalNewtonLineSearch::StepControlParams step_control_params(
      const ViscoplastUtils::LocalNewtonLineSearchParams& params)
  {
    return {.alpha_init = params.alpha_init, .max_iter = params.max_iter};
  }
}  // namespace

Core::Utils::LineSearch::RecoveryPolicy<
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType>
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::viscoplastic_error_recovery(
    const LocalNewtonLineSearchRecoveryParams& recovery_params)
{
  switch (recovery_params.strategy)
  {
    case RecoveryStrategy::abort:
      return LocalNewtonLineSearch::RecoveryPolicy<ErrorType>(
          LocalNewtonLineSearch::AlwaysAbort<ErrorType>{});
    case RecoveryStrategy::treat_as_too_high:
      return LocalNewtonLineSearch::RecoveryPolicy<ErrorType>(
          LocalNewtonLineSearch::AlwaysTreatAsTooHigh<ErrorType>{});
    case RecoveryStrategy::individual_contraction_factor:
    {
      const auto& factors = recovery_params.individual_contraction_factor;
      return LocalNewtonLineSearch::RecoveryPolicy<ErrorType>(
          LocalNewtonLineSearch::IndividualContractionFactor<ErrorType>({
              {ErrorType::overflow_error, factors.overflow_error},
              {ErrorType::negative_plastic_strain, factors.negative_plastic_strain},
              {ErrorType::failed_matrix_log_evaluation, factors.failed_matrix_log_evaluation},
              {ErrorType::failed_matrix_exp_evaluation, factors.failed_matrix_exp_evaluation},
          }));
    }
    default:
      FOUR_C_THROW("Unknown recovery strategy {}", EnumTools::enum_name(recovery_params.strategy));
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::Utils::LineSearch::MeritResult<
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType>
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_local_newton_merit(const double alpha,
    const Core::LinAlg::Matrix<10, 1>& current_sol, const Core::LinAlg::Matrix<10, 1>& dx,
    const InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationDeformationTensors&
        deftensors,
    const double temperature, const double last_plastic_strain, const double dt)
{
  Core::LinAlg::Matrix<10, 1> trial_sol(current_sol);
  trial_sol.update(alpha, dx, 1.0);

  auto error = ViscoplastUtils::ErrorType::no_errors;
  const Core::LinAlg::Matrix<10, 1> trial_residual =
      evaluate_local_newton_residual(deftensors.right_cg, temperature, trial_sol,
          last_plastic_strain, deftensors.elastic_predictor_inverse_plastic_defgrad, dt, error);

  if (error != ViscoplastUtils::ErrorType::no_errors) return {.value = 0.0, .error = error};

  return {.value = merit_from_residual(trial_residual), .error = std::nullopt};
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::globalized_local_newton_with_line_search(
    const InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationDeformationTensors&
        deftensors,
    const double temperature, const double last_plastic_strain, const double dt,
    Core::Utils::LineSearch::LineSearch<
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType>& line_search,
    InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
{
  ensure_error_free_evaluation(err_status);

  auto zero_result = []()
  { return Core::LinAlg::Matrix<10, 1>{Core::LinAlg::Initialization::zero}; };

  Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<10, 1> d(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<10, 1> residual(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<10, 1> temp10x1(Core::LinAlg::Initialization::zero);

  ViscoplastUtils::EvaluationAction eval_action{
      ViscoplastUtils::EvaluationAction::continue_current_iteration};

  local_newton_manager_.reset_iter();
  temp10x1 = determine_local_newton_init_estimate(dt, deftensors, last_plastic_strain, err_status);
  local_newton_manager_.save_init_estimate_and_reset_convergence_quantities(temp10x1);

  if (err_status != ViscoplastUtils::ErrorType::no_errors)
  {
    if (parameter()->use_local_substepping())
    {
      return zero_result();
    }

    FOUR_C_THROW(
        "{}", get_error_warning_info("Could not compute initial estimate for the local Newton!"));
  }

  while (true)
  {
    err_status = ViscoplastUtils::ErrorType::no_errors;

    residual = evaluate_local_newton_residual(deftensors.right_cg, temperature,
        local_newton_manager_.sol(), last_plastic_strain,
        deftensors.elastic_predictor_inverse_plastic_defgrad, dt, err_status);

    manage_evaluation(err_status, eval_action);
    switch (eval_action)
    {
      case ViscoplastUtils::EvaluationAction::continue_current_iteration:
      {
        break;
      }
      case ViscoplastUtils::EvaluationAction::continue_with_next_iteration:
      {
        local_newton_manager_.increment_iter();
        continue;
      }
      case ViscoplastUtils::EvaluationAction::exit_with_error:
      {
        return zero_result();
      }
      default:
      {
        FOUR_C_THROW("{}",
            get_error_warning_info(std::format("Invalid evaluation action {} for error status {}",
                EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status))));
      }
    }

    local_newton_manager_.set_residual_norm(residual);
    if (local_newton_manager_.is_local_newton_converged()) return local_newton_manager_.sol();

    if (local_newton_manager_.is_max_iter_reached())
    {
      err_status =
          InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_convergence_local_newton;

      if (parameter()->use_local_substepping())
      {
        return zero_result();
      }

      verify_local_newton_exit(err_status);
      return local_newton_manager_.sol();
    }

    if (local_newton_manager_.is_local_newton_stuck())
    {
      err_status =
          InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_convergence_local_newton;
      manage_evaluation(err_status, eval_action);
      switch (eval_action)
      {
        case ViscoplastUtils::EvaluationAction::continue_current_iteration:
        {
          break;
        }
        case ViscoplastUtils::EvaluationAction::continue_with_next_iteration:
        {
          local_newton_manager_.increment_iter();
          continue;
        }
        case ViscoplastUtils::EvaluationAction::exit_with_error:
        {
          return zero_result();
        }
        default:
        {
          FOUR_C_THROW("{}",
              get_error_warning_info(std::format("Invalid evaluation action {} for error status {}",
                  EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status))));
        }
      }
    }

    jacMat = evaluate_local_newton_jacobian(deftensors.right_cg, temperature,
        local_newton_manager_.sol(), last_plastic_strain,
        deftensors.elastic_predictor_inverse_plastic_defgrad, dt, err_status);

    manage_evaluation(err_status, eval_action);
    switch (eval_action)
    {
      case ViscoplastUtils::EvaluationAction::continue_current_iteration:
      {
        break;
      }
      case ViscoplastUtils::EvaluationAction::continue_with_next_iteration:
      {
        local_newton_manager_.increment_iter();
        continue;
      }
      case ViscoplastUtils::EvaluationAction::exit_with_error:
      {
        return zero_result();
      }
      default:
      {
        FOUR_C_THROW("{}",
            get_error_warning_info(std::format("Invalid evaluation action {} for error status {}",
                EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status))));
      }
    }

    const bool successful_solve = solve_local_newton_linear_system(residual, jacMat, d);
    if (!successful_solve)
    {
      err_status = ViscoplastUtils::ErrorType::failed_solution_linear_system_lnl;
      manage_evaluation(err_status, eval_action);
      switch (eval_action)
      {
        case ViscoplastUtils::EvaluationAction::continue_current_iteration:
        {
          break;
        }
        case ViscoplastUtils::EvaluationAction::continue_with_next_iteration:
        {
          local_newton_manager_.increment_iter();
          continue;
        }
        case ViscoplastUtils::EvaluationAction::exit_with_error:
        {
          return zero_result();
        }
        default:
        {
          FOUR_C_THROW("{}",
              get_error_warning_info(std::format("Invalid evaluation action {} for error status {}",
                  EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status))));
        }
      }
    }

    const Core::LinAlg::Matrix<10, 1> current_sol = local_newton_manager_.sol();
    const double merit_0 = merit_from_residual(residual);
    const double dmerit_da_0 = -2.0 * merit_0;

    auto merit =
        [this, &deftensors, temperature, last_plastic_strain, dt, &current_sol, &d](
            const double alpha) -> LocalNewtonLineSearch::MeritResult<ViscoplastUtils::ErrorType>
    {
      return this->evaluate_local_newton_merit(
          alpha, current_sol, d, deftensors, temperature, last_plastic_strain, dt);
    };

    const double alpha = line_search(dmerit_da_0, merit_0, std::move(merit));
    if (!std::isfinite(alpha) || alpha <= 0.0)
    {
      err_status =
          InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_convergence_local_newton;

      if (parameter()->use_local_substepping())
      {
        return zero_result();
      }

      verify_local_newton_exit(err_status);
      return local_newton_manager_.sol();
    }

    d.scale(alpha);
    local_newton_manager_.increment_solution_vector(d);
    local_newton_manager_.increment_iter();
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::run_local_newton_solve(
    const InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationDeformationTensors&
        deftensors,
    const double temperature, const double last_plastic_strain, const double dt,
    InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
{
  const auto& line_search_params = parameter()->local_newton_params().line_search;
  std::unique_ptr<LocalNewtonLineSearch::LineSearch<ViscoplastUtils::ErrorType>> line_search;
  switch (line_search_params.type)
  {
    case LocalNewtonLineSearch::LineSearchType::none:
    {
      // default value. If no line search is specified, the local Newton is run without line search
      return local_newton_loop(deftensors, temperature, last_plastic_strain, dt, err_status);
    }
    case LocalNewtonLineSearch::LineSearchType::armijo_backtracking:
    {
      using Condition = LocalNewtonLineSearch::ArmijoCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::Backtracking<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor,
          Condition(line_search_params.armijo,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::armijo_safeguarded_quadratic_backtracking:
    {
      using Condition = LocalNewtonLineSearch::ArmijoCondition<ViscoplastUtils::ErrorType>;
      line_search =
          std::make_unique<LocalNewtonLineSearch::SafeguardedQuadraticBacktracking<Condition,
              ViscoplastUtils::ErrorType>>(step_control_params(line_search_params),
              Condition(line_search_params.armijo,
                  viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::goldstein_simple_bisection:
    {
      using Condition = LocalNewtonLineSearch::GoldsteinCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::SimpleBisection<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          Condition(line_search_params.goldstein,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::weak_wolfe_simple_bisection:
    {
      using Condition = LocalNewtonLineSearch::WeakWolfeCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::SimpleBisection<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          Condition(line_search_params.wolfe,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::weak_wolfe_more_thuente:
    {
      using Condition = LocalNewtonLineSearch::WeakWolfeCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::MoreThuente<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.wolfe.c1,
          Condition(line_search_params.wolfe,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::strong_wolfe_simple_bisection:
    {
      using Condition = LocalNewtonLineSearch::StrongWolfeCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::SimpleBisection<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          Condition(line_search_params.wolfe,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::strong_wolfe_more_thuente:
    {
      using Condition = LocalNewtonLineSearch::StrongWolfeCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::MoreThuente<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.wolfe.c1,
          Condition(line_search_params.wolfe,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::dai_kou_simple_bisection:
    {
      using Condition = LocalNewtonLineSearch::DaiKouCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::SimpleBisection<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          Condition(line_search_params.dai_kou,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::grippo_lampariello_lucidi_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::GrippoLamparielloLucidiCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::Backtracking<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor,
          Condition(line_search_params.grippo_lampariello_lucidi,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::
        grippo_lampariello_lucidi_safeguarded_quadratic_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::GrippoLamparielloLucidiCondition<ViscoplastUtils::ErrorType>;
      line_search =
          std::make_unique<LocalNewtonLineSearch::SafeguardedQuadraticBacktracking<Condition,
              ViscoplastUtils::ErrorType>>(step_control_params(line_search_params),
              Condition(line_search_params.grippo_lampariello_lucidi,
                  viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::zhang_hager_nonmonotone_armijo_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::ZhangHagerNonmonotoneArmijoCondition<ViscoplastUtils::ErrorType>;
      line_search = std::make_unique<
          LocalNewtonLineSearch::Backtracking<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor,
          Condition(line_search_params.zhang_hager_nonmonotone,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::
        zhang_hager_nonmonotone_armijo_safeguarded_quadratic_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::ZhangHagerNonmonotoneArmijoCondition<ViscoplastUtils::ErrorType>;
      line_search =
          std::make_unique<LocalNewtonLineSearch::SafeguardedQuadraticBacktracking<Condition,
              ViscoplastUtils::ErrorType>>(step_control_params(line_search_params),
              Condition(line_search_params.zhang_hager_nonmonotone,
                  viscoplastic_error_recovery(line_search_params.recovery_policy)));
      break;
    }
    case LocalNewtonLineSearch::LineSearchType::hager_zhang:
    {
      line_search = std::make_unique<LocalNewtonLineSearch::HagerZhang<ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.hager_zhang,
          viscoplastic_error_recovery(line_search_params.recovery_policy));
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown Local Newton line-search type {}",
          EnumTools::enum_name(line_search_params.type));
    }
  }

  return globalized_local_newton_with_line_search(
      deftensors, temperature, last_plastic_strain, dt, *line_search, err_status);
}

FOUR_C_NAMESPACE_CLOSE
