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

  LineSearch::GoldsteinParams make_goldstein_params() { return {.rho = 0.1}; }

  LineSearch::WolfeParams make_wolfe_params() { return {.c1 = 1.0e-4, .c2 = 0.1}; }

  LineSearch::DaiKouParams make_dai_kou_params()
  {
    return {.epsilon = 100.0, .rho = 0.1, .sigma = 0.9, .eta_relative = 1.0};
  }

  LineSearch::GrippoLamparielloLucidiParams make_grippo_lampariello_lucidi_params()
  {
    return {.rho = 0.1, .max_history = 10};
  }

  LineSearch::ZhangHagerNonmonotoneParams make_zhang_hager_nonmonotone_params()
  {
    return {.rho = 0.1, .eta = 1.0};
  }

  LineSearch::HagerZhangParams make_hager_zhang_params()
  {
    return {.delta = 0.1, .sigma = 0.9, .epsilon = 1.0e-6, .theta = 0.5, .gamma = 0.66, .rho = 5.0};
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

  class ConditionWithoutMerit
  {
   public:
    LineSearch::BasicConditionResult operator()(
        double, double, const LineSearch::MeritFunction<TestEvaluationError>&, double) const
    {
      return {.status = LineSearch::TrialStatus::accepted, .recovery = std::nullopt, .data = {}};
    }
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

  static_assert(
      LineSearch::MeritLineSearchCondition<LineSearch::ArmijoCondition<TestEvaluationError>,
          TestEvaluationError>);
  static_assert(
      LineSearch::DMeritLineSearchCondition<LineSearch::StrongWolfeCondition<TestEvaluationError>,
          TestEvaluationError>);
  static_assert(
      LineSearch::DMeritLineSearchCondition<LineSearch::DaiKouCondition<TestEvaluationError>,
          TestEvaluationError>);
  static_assert(LineSearch::MeritLineSearchCondition<
      LineSearch::ZhangHagerNonmonotoneArmijoCondition<TestEvaluationError>, TestEvaluationError>);
  static_assert(!LineSearch::MeritLineSearchCondition<ConditionWithoutMerit, TestEvaluationError>);

  TEST(CoreUtilsLineSearchTest, DaiKouUsesInverseSquareSequencePerAcceptedIteration)
  {
    LineSearch::DaiKouCondition<TestEvaluationError> condition(
        make_dai_kou_params(), default_recovery());
    const auto constant_merit = [](const double value)
    {
      return [value](double)
      {
        return LineSearch::MeritResult<TestEvaluationError>{.value = value, .error = std::nullopt};
      };
    };

    EXPECT_EQ(
        condition(-1.0, 10.0, constant_merit(10.8), 1.0).status, LineSearch::TrialStatus::accepted);
    EXPECT_EQ(
        condition(-1.0, 10.8, constant_merit(10.9), 1.0).status, LineSearch::TrialStatus::accepted);

    EXPECT_EQ(condition(-1.0, 10.9, constant_merit(10.92), 1.0).status,
        LineSearch::TrialStatus::too_high);
    EXPECT_EQ(condition(-1.0, 10.9, constant_merit(10.91), 1.0).status,
        LineSearch::TrialStatus::accepted);
  }

  TEST(CoreUtilsLineSearchTest, ZhangHagerNonmonotoneConditionRetainsAveragedReferenceMerit)
  {
    using Condition = LineSearch::ZhangHagerNonmonotoneArmijoCondition<TestEvaluationError>;
    LineSearch::Backtracking<Condition, TestEvaluationError> line_search(make_step_control_params(),
        reduction_factor, Condition(make_zhang_hager_nonmonotone_params(), default_recovery()));
    const auto constant_merit = [](const double value)
    {
      return [value](double)
      {
        return LineSearch::MeritResult<TestEvaluationError>{.value = value, .error = std::nullopt};
      };
    };

    EXPECT_DOUBLE_EQ(line_search(-1.0, 10.0, constant_merit(8.0)), 1.0);
    EXPECT_DOUBLE_EQ(line_search(-1.0, 8.0, constant_merit(8.5)), 1.0);

    EXPECT_DOUBLE_EQ(line_search(-1.0, 8.5, constant_merit(8.75)), 0.5);
  }

  TEST(CoreUtilsLineSearchTest, GoldsteinConditionClassifiesTrialsAgainstBothBounds)
  {
    LineSearch::GoldsteinCondition<TestEvaluationError> condition(
        make_goldstein_params(), default_recovery());
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 3.0) * (alpha - 3.0), .error = std::nullopt};
    };

    EXPECT_EQ(condition(-6.0, 9.0, merit, 2.0).status, LineSearch::TrialStatus::accepted);
    EXPECT_EQ(condition(-6.0, 9.0, merit, 0.3).status, LineSearch::TrialStatus::too_low);
    EXPECT_EQ(condition(-6.0, 9.0, merit, 6.0).status, LineSearch::TrialStatus::too_high);
  }

  TEST(CoreUtilsLineSearchTest, SimpleBisectionDoublesFromTooLowUntilAccepted)
  {
    auto step_control = make_step_control_params();
    step_control.alpha_init = 0.2;
    using Condition = LineSearch::GoldsteinCondition<TestEvaluationError>;
    LineSearch::SimpleBisection<Condition, TestEvaluationError> line_search(
        step_control, Condition(make_goldstein_params(), default_recovery()));
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 3.0) * (alpha - 3.0), .error = std::nullopt};
    };

    EXPECT_DOUBLE_EQ(line_search(-6.0, 9.0, merit), 0.8);
  }

  TEST(CoreUtilsLineSearchTest, SimpleBisectionBisectsFromTooHighUntilAccepted)
  {
    auto step_control = make_step_control_params();
    step_control.alpha_init = 6.0;
    using Condition = LineSearch::GoldsteinCondition<TestEvaluationError>;
    LineSearch::SimpleBisection<Condition, TestEvaluationError> line_search(
        step_control, Condition(make_goldstein_params(), default_recovery()));
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 3.0) * (alpha - 3.0), .error = std::nullopt};
    };

    EXPECT_DOUBLE_EQ(line_search(-6.0, 9.0, merit), 3.0);
  }

  TEST(CoreUtilsLineSearchTest, WeakWolfeConditionClassifiesTrialsBySufficientDecreaseAndCurvature)
  {
    LineSearch::WeakWolfeCondition<TestEvaluationError> condition(
        make_wolfe_params(), default_recovery());
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 4.0) * (alpha - 4.0), .error = std::nullopt};
    };

    EXPECT_EQ(condition(-8.0, 16.0, merit, 4.0).status, LineSearch::TrialStatus::accepted);
    EXPECT_EQ(condition(-8.0, 16.0, merit, 1.0).status, LineSearch::TrialStatus::too_low);
    EXPECT_EQ(condition(-8.0, 16.0, merit, 20.0).status, LineSearch::TrialStatus::too_high);
  }

  TEST(CoreUtilsLineSearchTest, HagerZhangRejectsNonDescentDirection)
  {
    LineSearch::HagerZhang<TestEvaluationError> line_search(
        make_step_control_params(), make_hager_zhang_params(), default_recovery());
    const auto merit = [](const double alpha)
    { return LineSearch::MeritResult<TestEvaluationError>{.value = alpha, .error = std::nullopt}; };

    EXPECT_DOUBLE_EQ(line_search(1.0, 0.0, merit), 0.0);
  }

  TEST(CoreUtilsLineSearchTest, HagerZhangSatisfiesWolfeConditionsAtReturnedStep)
  {
    auto step_control = make_step_control_params();
    step_control.max_iter = 30;
    const auto hager_zhang_params = make_hager_zhang_params();
    LineSearch::HagerZhang<TestEvaluationError> line_search(
        step_control, hager_zhang_params, default_recovery());
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 4.0) * (alpha - 4.0), .error = std::nullopt};
    };

    const double alpha = line_search(-8.0, 16.0, merit);
    ASSERT_GT(alpha, 0.0);

    const double merit_alpha = (alpha - 4.0) * (alpha - 4.0);
    const double derivative_alpha = 2.0 * (alpha - 4.0);
    EXPECT_LE(merit_alpha,
        16.0 + hager_zhang_params.delta * alpha * -8.0 + hager_zhang_params.epsilon * 16.0);
    EXPECT_GE(derivative_alpha, hager_zhang_params.sigma * -8.0);
  }

  TEST(CoreUtilsLineSearchTest, MoreThuenteInterpolatesWhenInitialStepIsTooLow)
  {
    auto step_control = make_step_control_params();
    step_control.max_iter = 20;
    const auto wolfe_params = make_wolfe_params();
    using Condition = LineSearch::StrongWolfeCondition<TestEvaluationError>;
    LineSearch::MoreThuente<Condition, TestEvaluationError> line_search(
        step_control, wolfe_params.c1, Condition(wolfe_params, default_recovery()));
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 4.0) * (alpha - 4.0), .error = std::nullopt};
    };

    EXPECT_NEAR(line_search(-8.0, 16.0, merit), 4.0, 1.0e-6);
  }

  TEST(CoreUtilsLineSearchTest, StrongWolfeSimpleBisectionDoublesFromTooLowUntilAccepted)
  {
    auto step_control = make_step_control_params();
    step_control.alpha_init = 3.2;
    step_control.max_iter = 10;
    using Condition = LineSearch::StrongWolfeCondition<TestEvaluationError>;
    LineSearch::SimpleBisection<Condition, TestEvaluationError> line_search(
        step_control, Condition(make_wolfe_params(), default_recovery()));
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 4.0) * (alpha - 4.0), .error = std::nullopt};
    };

    EXPECT_NEAR(line_search(-8.0, 16.0, merit), 4.0, 1.0e-9);
  }

  TEST(CoreUtilsLineSearchTest, MoreThuenteWithWeakWolfeConditionSatisfiesWeakWolfeAtReturnedStep)
  {
    auto step_control = make_step_control_params();
    step_control.max_iter = 20;
    const auto wolfe_params = make_wolfe_params();
    using Condition = LineSearch::WeakWolfeCondition<TestEvaluationError>;
    LineSearch::MoreThuente<Condition, TestEvaluationError> line_search(
        step_control, wolfe_params.c1, Condition(wolfe_params, default_recovery()));
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 4.0) * (alpha - 4.0), .error = std::nullopt};
    };

    const double alpha = line_search(-8.0, 16.0, merit);
    ASSERT_GT(alpha, 0.0);

    const double merit_alpha = (alpha - 4.0) * (alpha - 4.0);
    const double derivative_alpha = 2.0 * (alpha - 4.0);
    EXPECT_LE(merit_alpha, 16.0 + wolfe_params.c1 * alpha * -8.0);
    EXPECT_GE(derivative_alpha, wolfe_params.c2 * -8.0);
  }

  TEST(CoreUtilsLineSearchTest, SafeguardedQuadraticBacktrackingFindsQuadraticMinimizer)
  {
    using Condition = LineSearch::ArmijoCondition<TestEvaluationError>;
    LineSearch::SafeguardedQuadraticBacktracking<Condition, TestEvaluationError> line_search(
        make_step_control_params(), Condition(make_armijo_params(), default_recovery()));
    std::vector<double> evaluated_alphas;
    const auto merit = [&evaluated_alphas](const double alpha)
    {
      evaluated_alphas.push_back(alpha);
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 0.25) * (alpha - 0.25), .error = std::nullopt};
    };

    EXPECT_DOUBLE_EQ(line_search(-0.5, 0.0625, merit), 0.25);
    ASSERT_EQ(evaluated_alphas.size(), 2);
    EXPECT_DOUBLE_EQ(evaluated_alphas[0], 1.0);
    EXPECT_DOUBLE_EQ(evaluated_alphas[1], 0.25);
  }

  TEST(CoreUtilsLineSearchTest, NonmonotoneConditionsSupportSafeguardedQuadraticBacktracking)
  {
    using GllCondition = LineSearch::GrippoLamparielloLucidiCondition<TestEvaluationError>;
    LineSearch::SafeguardedQuadraticBacktracking<GllCondition, TestEvaluationError> gll(
        make_step_control_params(),
        GllCondition(make_grippo_lampariello_lucidi_params(), default_recovery()));
    using ZhangHagerNonmonotoneCondition =
        LineSearch::ZhangHagerNonmonotoneArmijoCondition<TestEvaluationError>;
    LineSearch::SafeguardedQuadraticBacktracking<ZhangHagerNonmonotoneCondition,
        TestEvaluationError>
        zhang_hager_nonmonotone(make_step_control_params(),
            ZhangHagerNonmonotoneCondition(
                make_zhang_hager_nonmonotone_params(), default_recovery()));
    const auto merit = [](const double alpha)
    {
      return LineSearch::MeritResult<TestEvaluationError>{
          .value = (alpha - 0.25) * (alpha - 0.25), .error = std::nullopt};
    };

    EXPECT_DOUBLE_EQ(gll(-0.5, 0.0625, merit), 0.25);
    EXPECT_DOUBLE_EQ(zhang_hager_nonmonotone(-0.5, 0.0625, merit), 0.25);
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
