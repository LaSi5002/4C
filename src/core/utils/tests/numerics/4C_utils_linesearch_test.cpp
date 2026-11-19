// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_utils_linesearch.hpp"

#include <cmath>
#include <limits>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  namespace LineSearch = Core::Utils::LineSearch;

  LineSearch::StepControlParams make_step_control_params()
  {
    return {.alpha_init = 1.0, .max_iter = 5};
  }

  LineSearch::ArmijoParams make_armijo_params() { return {.c1 = 1.0e-4}; }

  LineSearch::GrippoLamparielloLucidiParams make_grippo_lampariello_lucidi_params()
  {
    return {.rho = 0.1, .max_history = 10};
  }

  constexpr double reduction_factor = 0.5;

  enum class TestEvaluationError
  {
    error_a,
    error_b,
    fatal
  };

  LineSearch::RecoveryPolicy<TestEvaluationError> default_recovery()
  {
    return LineSearch::RecoveryPolicy<TestEvaluationError>(
        LineSearch::IndividualContractionFactor<TestEvaluationError>({
            {TestEvaluationError::error_a, 0.25},
            {TestEvaluationError::error_b, 0.5},
        }));
  }

  class MockErrorRecovery
  {
   public:
    explicit MockErrorRecovery(std::map<TestEvaluationError, LineSearch::RecoveryAction> recoveries)
        : recoveries_(std::move(recoveries))
    {
    }

    LineSearch::RecoveryAction operator()(const TestEvaluationError error) const
    {
      const auto recovery_entry = recoveries_.find(error);
      return recovery_entry != recoveries_.end() ? recovery_entry->second
                                                 : LineSearch::AbortLineSearch{};
    }

   private:
    std::map<TestEvaluationError, LineSearch::RecoveryAction> recoveries_;
  };

  MockErrorRecovery default_mock_recovery()
  {
    return MockErrorRecovery({
        {TestEvaluationError::error_a, LineSearch::ContractStep{.factor = 0.25}},
        {TestEvaluationError::error_b, LineSearch::ContractStep{.factor = 0.5}},
        {TestEvaluationError::fatal, LineSearch::AbortLineSearch{}},
    });
  }

  class PrescribedMeritCondition
  {
   public:
    explicit PrescribedMeritCondition(MockErrorRecovery recovery_policy)
        : recovery_policy_(std::move(recovery_policy))
    {
    }

    LineSearch::MeritConditionResult operator()(double, double,
        const LineSearch::MeritFunction<TestEvaluationError>& merit, const double alpha) const
    {
      const LineSearch::MeritResult<TestEvaluationError> merit_alpha = merit(alpha);
      return {.status = alpha <= acceptance_threshold ? LineSearch::TrialStatus::accepted
                                                      : LineSearch::TrialStatus::too_high,
          .recovery = merit_alpha.error ? std::optional(recovery_policy_(*merit_alpha.error))
                                        : std::nullopt,
          .data = {.merit = merit_alpha.value}};
    }

    static inline double acceptance_threshold = 0.0;

   private:
    MockErrorRecovery recovery_policy_;
  };

  class TooLowMeritCondition
  {
   public:
    explicit TooLowMeritCondition(MockErrorRecovery recovery_policy)
        : recovery_policy_(std::move(recovery_policy))
    {
    }

    LineSearch::MeritConditionResult operator()(double, double,
        const LineSearch::MeritFunction<TestEvaluationError>& merit, const double alpha) const
    {
      const LineSearch::MeritResult<TestEvaluationError> merit_alpha = merit(alpha);
      return {.status = LineSearch::TrialStatus::too_low,
          .recovery = merit_alpha.error ? std::optional(recovery_policy_(*merit_alpha.error))
                                        : std::nullopt,
          .data = {.merit = merit_alpha.value}};
    }

   private:
    MockErrorRecovery recovery_policy_;
  };

  TEST(CoreUtilsLineSearchTest, GoldenSectionSearchRejectsNonDescentDirection)
  {
    LineSearch::GoldenSectionSearch<TestEvaluationError> line_search(
        make_step_control_params(), default_recovery());
    const auto merit = [](const double alpha)
    { return LineSearch::MeritResult<TestEvaluationError>{.value = alpha, .error = std::nullopt}; };

    EXPECT_DOUBLE_EQ(line_search(1.0, 0.0, merit), 0.0);
  }

  TEST(CoreUtilsLineSearchTest, GoldenSectionSearchFindsExactMinimizer)
  {
    auto step_control = make_step_control_params();
    step_control.max_iter = 50;
    LineSearch::GoldenSectionSearch<TestEvaluationError> line_search(
        step_control, default_recovery());
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 4.0) * (alpha - 4.0), .error = std::nullopt};
    };

    EXPECT_NEAR(line_search(-8.0, 16.0, merit), 4.0, 1.0e-6);
  }

  TEST(CoreUtilsLineSearchTest, BacktrackingUsesErrorSpecificContractionFactors)
  {
    auto step_control = make_step_control_params();
    step_control.max_iter = 3;
    const MockErrorRecovery recover({
        {TestEvaluationError::error_a, LineSearch::ContractStep{.factor = 0.25}},
        {TestEvaluationError::error_b, LineSearch::ContractStep{.factor = 0.5}},
        {TestEvaluationError::fatal, LineSearch::AbortLineSearch{}},
    });
    LineSearch::Backtracking<PrescribedMeritCondition, TestEvaluationError> line_search(
        step_control, reduction_factor, PrescribedMeritCondition(recover));
    PrescribedMeritCondition::acceptance_threshold = 0.125;
    std::vector<double> evaluated_alphas;
    const auto merit = [&evaluated_alphas](const double alpha)
    {
      evaluated_alphas.push_back(alpha);
      if (alpha == 1.0)
        return LineSearch::MeritResult<TestEvaluationError>{
            .value = 0.0, .error = TestEvaluationError::error_a};
      if (alpha == 0.25)
        return LineSearch::MeritResult<TestEvaluationError>{
            .value = 0.0, .error = TestEvaluationError::error_b};
      return LineSearch::MeritResult<TestEvaluationError>{.value = 0.0, .error = std::nullopt};
    };

    EXPECT_DOUBLE_EQ(line_search(-1.0, 1.0, merit), 0.125);
    EXPECT_EQ(evaluated_alphas, (std::vector<double>{1.0, 0.25, 0.125}));
  }

  TEST(CoreUtilsLineSearchTest, FixedBacktrackingStopsWhenTrialIsTooLow)
  {
    LineSearch::Backtracking<TooLowMeritCondition, TestEvaluationError> line_search(
        make_step_control_params(), reduction_factor,
        TooLowMeritCondition(default_mock_recovery()));
    int evaluation_count = 0;
    const auto merit = [&evaluation_count](double)
    {
      ++evaluation_count;
      return LineSearch::MeritResult<TestEvaluationError>{.value = 1.0, .error = std::nullopt};
    };

    EXPECT_DOUBLE_EQ(line_search(-1.0, 1.0, merit), 0.0);
    EXPECT_EQ(evaluation_count, 1);
  }

  TEST(CoreUtilsLineSearchTest, ConditionWithAlwaysAbortPolicyAbortsOnAnyError)
  {
    using Condition = LineSearch::ArmijoCondition<TestEvaluationError>;
    LineSearch::Backtracking<Condition, TestEvaluationError> line_search(make_step_control_params(),
        reduction_factor,
        Condition(make_armijo_params(), LineSearch::RecoveryPolicy<TestEvaluationError>(
                                            LineSearch::AlwaysAbort<TestEvaluationError>{})));
    const auto merit = [](double)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = 0.0, .error = TestEvaluationError::fatal};
    };

    EXPECT_DOUBLE_EQ(line_search(-1.0, 1.0, merit), 0.0);
  }
}  // namespace
FOUR_C_NAMESPACE_CLOSE
