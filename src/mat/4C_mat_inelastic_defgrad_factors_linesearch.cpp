// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace
{
  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;
  namespace LocalNewtonLineSearch = Core::Utils::LineSearch;

  [[nodiscard]] double merit_from_residual(const Core::LinAlg::Matrix<10, 1>& residual)
  {
    return 0.5 * residual.dot(residual);
  }

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
std::optional<double>
Mat::InelasticDefgradTransvIsotropElastViscoplast::determine_line_search_step_length(
    const InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationInput&
        local_integration_input,
    const Core::LinAlg::Matrix<10, 1>& residual, const Core::LinAlg::Matrix<10, 1>& dx,
    std::optional<Core::LinAlg::Matrix<10, 1>>& reusable_residual)
{
  const Core::LinAlg::Matrix<10, 1> current_sol = local_newton_manager_.sol();
  const double merit_0 = merit_from_residual(residual);
  const double dmerit_da_0 = -2.0 * merit_0;

  double last_evaluated_alpha = std::numeric_limits<double>::quiet_NaN();
  Core::LinAlg::Matrix<10, 1> last_evaluated_residual(Core::LinAlg::Initialization::zero);

  auto merit =
      [this, &local_integration_input, &current_sol, &dx, &last_evaluated_alpha,
          &last_evaluated_residual](
          const double alpha) -> LocalNewtonLineSearch::MeritResult<ViscoplastUtils::ErrorType>
  {
    Core::LinAlg::Matrix<10, 1> trial_residual(Core::LinAlg::Initialization::zero);
    const auto result = this->evaluate_local_newton_merit(
        alpha, current_sol, dx, local_integration_input, &trial_residual);
    if (!result.error.has_value())
    {
      last_evaluated_alpha = alpha;
      last_evaluated_residual = trial_residual;
    }
    else
    {
      last_evaluated_alpha = std::numeric_limits<double>::quiet_NaN();
    }
    return result;
  };

  const double alpha = (*line_search_)(dmerit_da_0, merit_0, merit);
  if (!std::isfinite(alpha) || alpha < 0.0) return std::nullopt;

  if (last_evaluated_alpha == alpha) reusable_residual = last_evaluated_residual;

  return alpha;
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
    case LocalNewtonLineSearch::LineSearchType::error_adaptive_newton:
    {
      return std::make_unique<
          LocalNewtonLineSearch::ErrorAdaptiveNewton<ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor);
    }
    case LocalNewtonLineSearch::LineSearchType::armijo_backtracking:
    {
      using Condition = LocalNewtonLineSearch::ArmijoCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::Backtracking<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor,
          Condition(line_search_params.armijo,
              viscoplastic_error_recovery(line_search_params.recovery_policy)),
          line_search_params.backtracking.max_line_search_iterations);
    }
    case LocalNewtonLineSearch::LineSearchType::grippo_lampariello_lucidi_backtracking:
    {
      using Condition =
          LocalNewtonLineSearch::GrippoLamparielloLucidiCondition<ViscoplastUtils::ErrorType>;
      return std::make_unique<
          LocalNewtonLineSearch::Backtracking<Condition, ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params), line_search_params.reduction_factor,
          Condition(line_search_params.grippo_lampariello_lucidi,
              viscoplastic_error_recovery(line_search_params.recovery_policy)),
          line_search_params.backtracking.max_line_search_iterations);
    }
    case LocalNewtonLineSearch::LineSearchType::golden_section:
    {
      return std::make_unique<
          LocalNewtonLineSearch::GoldenSectionSearch<ViscoplastUtils::ErrorType>>(
          step_control_params(line_search_params),
          viscoplastic_error_recovery(line_search_params.recovery_policy),
          line_search_params.golden_section.min_alpha,
          line_search_params.golden_section.interval_tol);
    }
    default:
    {
      FOUR_C_THROW("Unknown Local Newton line-search type {}",
          EnumTools::enum_name(line_search_params.type));
    }
  }
}


FOUR_C_NAMESPACE_CLOSE
