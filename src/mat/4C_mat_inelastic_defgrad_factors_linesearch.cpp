// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors.hpp"
#include "4C_mat_inelastic_defgrad_factors_merit_export.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace
{
  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;
  namespace LocalNewtonLineSearch = Core::Utils::LineSearch;
  using MeritExportDebug::MeritCurveExporter;

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
    const InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationInput&
        local_integration_input,
    Core::LinAlg::Matrix<10, 1>* residual_out)
{
  Core::LinAlg::Matrix<10, 1> trial_sol(current_sol);
  trial_sol.update(alpha, dx, 1.0);

  auto error = ViscoplastUtils::ErrorType::no_errors;
  const Core::LinAlg::Matrix<10, 1> trial_residual =
      evaluate_local_newton_residual(local_integration_input, trial_sol, error);

  if (error != ViscoplastUtils::ErrorType::no_errors) return {.value = 0.0, .error = error};

  if (residual_out != nullptr) *residual_out = trial_residual;

  return {.value = merit_from_residual(trial_residual), .error = std::nullopt};
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::globalized_local_newton_with_line_search(
    const InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationInput&
        local_integration_input,
    Core::Utils::LineSearch::LineSearch<
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType>& line_search,
    InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
{
  ensure_error_free_evaluation(err_status);

  const int merit_curve_trajectory_id = MeritCurveExporter::instance().next_trajectory_id();

  auto zero_result = []()
  { return Core::LinAlg::Matrix<10, 1>{Core::LinAlg::Initialization::zero}; };

  Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<10, 1> d(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<10, 1> residual(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<10, 1> temp10x1(Core::LinAlg::Initialization::zero);

  std::optional<Core::LinAlg::Matrix<10, 1>> reusable_residual;

  ViscoplastUtils::EvaluationAction eval_action{
      ViscoplastUtils::EvaluationAction::continue_current_iteration};

  local_newton_manager_.reset_iter();
  temp10x1 = determine_local_newton_init_estimate(local_integration_input, err_status);
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

    if (reusable_residual.has_value())
    {
      residual = *reusable_residual;
      reusable_residual.reset();
    }
    else
    {
      residual = evaluate_local_newton_residual(
          local_integration_input, local_newton_manager_.sol(), err_status);

      manage_evaluation(err_status, local_integration_input, eval_action);
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

      if (adaptive_estimate_interp_manager_.has_value())
      {
        manage_evaluation(err_status, local_integration_input, eval_action);
        switch (eval_action)
        {
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
                get_error_warning_info(
                    std::format("Invalid evaluation action {} for error status {}",
                        EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status))));
          }
        }
      }
      else
      {
        verify_local_newton_exit(err_status);
        return local_newton_manager_.sol();
      }
    }

    if (local_newton_manager_.is_local_newton_stuck())
    {
      err_status =
          InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_convergence_local_newton;
      manage_evaluation(err_status, local_integration_input, eval_action);
      switch (eval_action)
      {
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

    jacMat = evaluate_local_newton_jacobian(
        local_integration_input, local_newton_manager_.sol(), err_status);

    manage_evaluation(err_status, local_integration_input, eval_action);
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
      manage_evaluation(err_status, local_integration_input, eval_action);
      switch (eval_action)
      {
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

    double last_evaluated_alpha = std::numeric_limits<double>::quiet_NaN();
    Core::LinAlg::Matrix<10, 1> last_evaluated_residual(Core::LinAlg::Initialization::zero);

    auto merit =
        [this, &local_integration_input, &current_sol, &d, &last_evaluated_alpha,
            &last_evaluated_residual](
            const double alpha) -> LocalNewtonLineSearch::MeritResult<ViscoplastUtils::ErrorType>
    {
      Core::LinAlg::Matrix<10, 1> trial_residual(Core::LinAlg::Initialization::zero);
      const auto result = this->evaluate_local_newton_merit(
          alpha, current_sol, d, local_integration_input, &trial_residual);
      if (!result.error.has_value())
      {
        last_evaluated_alpha = alpha;
        last_evaluated_residual = trial_residual;
      }
      return result;
    };

    const double alpha = line_search(dmerit_da_0, merit_0, merit);

    MeritCurveExporter::instance().maybe_export(
        merit, merit_curve_trajectory_id, local_newton_manager_.iter(), alpha);

    if (!std::isfinite(alpha) || alpha <= 0.0)
    {
      err_status =
          InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_convergence_local_newton;
      manage_evaluation(err_status, local_integration_input, eval_action);
      switch (eval_action)
      {
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

    d.scale(alpha);
    local_newton_manager_.increment_solution_vector(d);
    local_newton_manager_.increment_iter();

    if (last_evaluated_alpha == alpha)
    {
      reusable_residual = last_evaluated_residual;
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::run_local_newton_solve(
    const InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationInput&
        local_integration_input,
    InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
{
  if (!line_search_) return local_newton_loop(local_integration_input, err_status);

  return globalized_local_newton_with_line_search(
      local_integration_input, *line_search_, err_status);
}

std::unique_ptr<Core::Utils::LineSearch::LineSearch<
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType>>
Mat::InelasticDefgradTransvIsotropElastViscoplast::build_line_search() const
{
  const auto& line_search_params = parameter()->local_newton_params().line_search;
  switch (line_search_params.type)
  {
    case LocalNewtonLineSearch::LineSearchType::none:
    {
      return nullptr;
    }
    case LocalNewtonLineSearch::LineSearchType::armijo_backtracking:
    {
      using Condition = LocalNewtonLineSearch::ArmijoCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::Backtracking<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor,
          Condition(line_search_params.armijo,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::armijo_safeguarded_quadratic_backtracking:
    {
      using Condition = LocalNewtonLineSearch::ArmijoCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<LocalNewtonLineSearch::SafeguardedQuadraticBacktracking<Condition,
          ViscoplastUtils::ErrorType>>(step_control_params(line_search_params),
          Condition(line_search_params.armijo,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::goldstein_simple_bisection:
    {
      using Condition = LocalNewtonLineSearch::GoldsteinCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::SimpleBisection<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          Condition(line_search_params.goldstein,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::weak_wolfe_simple_bisection:
    {
      using Condition = LocalNewtonLineSearch::WeakWolfeCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::SimpleBisection<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          Condition(line_search_params.wolfe,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::weak_wolfe_more_thuente:
    {
      using Condition = LocalNewtonLineSearch::WeakWolfeCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::MoreThuente<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.wolfe.c1,
          Condition(line_search_params.wolfe,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::strong_wolfe_simple_bisection:
    {
      using Condition = LocalNewtonLineSearch::StrongWolfeCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::SimpleBisection<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          Condition(line_search_params.wolfe,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::strong_wolfe_more_thuente:
    {
      using Condition = LocalNewtonLineSearch::StrongWolfeCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::MoreThuente<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.wolfe.c1,
          Condition(line_search_params.wolfe,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::dai_kou_simple_bisection:
    {
      using Condition = LocalNewtonLineSearch::DaiKouCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::SimpleBisection<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          Condition(line_search_params.dai_kou,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::grippo_lampariello_lucidi_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::GrippoLamparielloLucidiCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::Backtracking<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor,
          Condition(line_search_params.grippo_lampariello_lucidi,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::
        grippo_lampariello_lucidi_safeguarded_quadratic_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::GrippoLamparielloLucidiCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<LocalNewtonLineSearch::SafeguardedQuadraticBacktracking<Condition,
          ViscoplastUtils::ErrorType>>(step_control_params(line_search_params),
          Condition(line_search_params.grippo_lampariello_lucidi,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::zhang_hager_nonmonotone_armijo_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::ZhangHagerNonmonotoneArmijoCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::Backtracking<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor,
          Condition(line_search_params.zhang_hager_nonmonotone,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::
        zhang_hager_nonmonotone_armijo_safeguarded_quadratic_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::ZhangHagerNonmonotoneArmijoCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<LocalNewtonLineSearch::SafeguardedQuadraticBacktracking<Condition,
          ViscoplastUtils::ErrorType>>(step_control_params(line_search_params),
          Condition(line_search_params.zhang_hager_nonmonotone,
              viscoplastic_error_recovery(line_search_params.recovery_policy)));
    }
    case LocalNewtonLineSearch::LineSearchType::hager_zhang:
    {
      return std::make_unique<LocalNewtonLineSearch::HagerZhang<ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.hager_zhang,
          viscoplastic_error_recovery(line_search_params.recovery_policy));
    }
    case LocalNewtonLineSearch::LineSearchType::golden_section:
    {
      return std::make_unique<
          LocalNewtonLineSearch::GoldenSectionSearch<ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          viscoplastic_error_recovery(line_search_params.recovery_policy));
    }
    default:
    {
      FOUR_C_THROW("Unknown Local Newton line-search type {}",
          EnumTools::enum_name(line_search_params.type));
    }
  }
}


FOUR_C_NAMESPACE_CLOSE
