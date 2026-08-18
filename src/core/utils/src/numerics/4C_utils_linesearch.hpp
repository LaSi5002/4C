// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_LINESEARCH_HPP
#define FOUR_C_UTILS_LINESEARCH_HPP

#include "4C_config.hpp"

#include "4C_utils_linesearch_params.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

FOUR_C_NAMESPACE_OPEN

//! Generic line-search toolkit for finding a step length alpha that satisfies
//! some condition on a merit function phi(alpha).
//!
//! An example of a merit function would be the a norm of a residual of a nonlinear system of
//! equations. phi(alpha) = || r(x + alpha * d) ||, r being the residual, x the current solution,
//! and d the search direction.
//!
//! Different algorithms from nonlinear optimization literature are implemented.
//!
//! @see N. Andrei, Modern Numerical Nonlinear Optimization,
//!      Springer, 2022, Sec. 2.5.
namespace Core::Utils::LineSearch
{
  struct ContractStep
  {
    double factor;
  };

  struct TreatAsTooHigh
  {
  };

  struct AbortLineSearch
  {
  };

  using RecoveryAction = std::variant<ContractStep, TreatAsTooHigh, AbortLineSearch>;

  /**
   * A RecoveryPolicy that returns AbortLineSearch for any error.
   */
  template <typename Error>
    requires std::is_enum_v<Error>
  class AlwaysAbort
  {
   public:
    [[nodiscard]] RecoveryAction operator()(const Error) const { return AbortLineSearch{}; }
  };

  /**
   * A RecoveryPolicy that returns TreatAsTooHigh for any error.
   */
  template <typename Error>
    requires std::is_enum_v<Error>
  class AlwaysTreatAsTooHigh
  {
   public:
    [[nodiscard]] RecoveryAction operator()(const Error) const { return TreatAsTooHigh{}; }
  };

  /**
   * A RecoveryPolicy alternative that returns a ContractStep with a user-specified contraction
   * factor for each error.
   */
  template <typename Error>
    requires std::is_enum_v<Error>
  class IndividualContractionFactor
  {
   public:
    explicit IndividualContractionFactor(std::map<Error, double> factors)
        : factors_(std::move(factors))
    {
    }

    [[nodiscard]] RecoveryAction operator()(const Error error) const
    {
      const auto factor = factors_.find(error);
      if (factor == factors_.end()) return AbortLineSearch{};
      return ContractStep{.factor = factor->second};
    }

   private:
    std::map<Error, double> factors_;
  };

  /**
   * Wrapper for the RecoveryPolicy alternatives above
   * which simplifies calling operator() on the selected policy.
   */
  template <typename Error>
    requires std::is_enum_v<Error>
  class RecoveryPolicy
  {
   public:
    using Alternative = std::variant<AlwaysAbort<Error>, AlwaysTreatAsTooHigh<Error>,
        IndividualContractionFactor<Error>>;

    RecoveryPolicy() = default;
    explicit RecoveryPolicy(Alternative policy) : policy_(std::move(policy)) {}

    [[nodiscard]] RecoveryAction operator()(const Error error) const
    {
      return std::visit([error](const auto& policy) { return policy(error); }, policy_);
    }

   private:
    Alternative policy_;
  };

  /**
   * Defines a generic result type for evaluating a merit function, which
   * may fail with different errors. Used in defining the LineSearch interface.
   */
  template <typename Error>
    requires std::is_enum_v<Error>
  struct MeritResult
  {
    double value;
    std::optional<Error> error;
  };

  struct NoConditionData
  {
  };

  struct MeritConditionData
  {
    double merit;
  };

  struct DMeritConditionData
  {
    double merit;
    double dmerit_da;
  };

  enum class TrialStatus
  {
    accepted,
    too_low,
    too_high
  };

  template <typename Data = NoConditionData>
  struct ConditionResult
  {
    TrialStatus status;
    std::optional<RecoveryAction> recovery;
    Data data;
  };

  using BasicConditionResult = ConditionResult<>;
  using MeritConditionResult = ConditionResult<MeritConditionData>;
  using DMeritConditionResult = ConditionResult<DMeritConditionData>;

  template <typename Error>
  using MeritFunction = std::function<MeritResult<Error>(double)>;

  template <typename Data>
  [[nodiscard]] ConditionResult<Data> invalid_alpha_result(const double alpha)
  {
    return {.status = std::isfinite(alpha) ? TrialStatus::too_low : TrialStatus::too_high,
        .recovery = std::nullopt,
        .data = Data{}};
  }

  /**
   * Base class for line-search algorithms finding a step length alpha
   * along some step direction that satisfies a given condition on a merit function phi(alpha).
   *
   * An example of a merit function would be the a norm of a residual of a nonlinear system of
   * equations. phi(alpha) = || r(x + alpha * d) ||, r being the residual, x the current solution,
   * and d the search direction.
   *
   * Can handle different errors using a RecoveryPolicy, which is passed to the Condition
   * that evaluates whether a given step length is acceptable.
   */
  template <typename Error>
    requires std::is_enum_v<Error>
  class LineSearch
  {
   public:
    explicit LineSearch(const StepControlParams& params) : params_(params) {}
    virtual ~LineSearch() = default;

    LineSearch(const LineSearch&) = delete;
    LineSearch& operator=(const LineSearch&) = delete;
    LineSearch(LineSearch&&) = delete;
    LineSearch& operator=(LineSearch&&) = delete;

    [[nodiscard]] virtual double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) = 0;

   protected:
    StepControlParams params_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class ArmijoCondition
  {
   public:
    explicit ArmijoCondition(
        ArmijoParams params, RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{})
        : params_(params), recovery_policy_(std::move(recovery_policy))
    {
    }

    [[nodiscard]] MeritConditionResult operator()(const double dmerit_da_0, const double merit_0,
        const MeritFunction<Error>& merit, const double alpha) const;

   private:
    ArmijoParams params_;
    RecoveryPolicy<Error> recovery_policy_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class GoldsteinCondition
  {
   public:
    explicit GoldsteinCondition(
        GoldsteinParams params, RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{})
        : params_(params), recovery_policy_(std::move(recovery_policy))
    {
    }

    [[nodiscard]] MeritConditionResult operator()(const double dmerit_da_0, const double merit_0,
        const MeritFunction<Error>& merit, const double alpha) const;

   private:
    GoldsteinParams params_;
    RecoveryPolicy<Error> recovery_policy_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class WeakWolfeCondition
  {
   public:
    explicit WeakWolfeCondition(
        WolfeParams params, RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{})
        : params_(params), recovery_policy_(std::move(recovery_policy))
    {
    }

    [[nodiscard]] DMeritConditionResult operator()(const double dmerit_da_0, const double merit_0,
        const MeritFunction<Error>& merit, const double alpha) const;

   private:
    WolfeParams params_;
    RecoveryPolicy<Error> recovery_policy_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class StrongWolfeCondition
  {
   public:
    explicit StrongWolfeCondition(
        WolfeParams params, RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{})
        : params_(params), recovery_policy_(std::move(recovery_policy))
    {
    }

    [[nodiscard]] DMeritConditionResult operator()(const double dmerit_da_0, const double merit_0,
        const MeritFunction<Error>& merit, const double alpha) const;

   private:
    WolfeParams params_;
    RecoveryPolicy<Error> recovery_policy_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class DaiKouCondition
  {
   public:
    explicit DaiKouCondition(
        DaiKouParams params, RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{})
        : params_(params), recovery_policy_(std::move(recovery_policy))
    {
    }

    [[nodiscard]] DMeritConditionResult operator()(const double dmerit_da_0, const double merit_0,
        const MeritFunction<Error>& merit, const double alpha);

   private:
    DaiKouParams params_;
    RecoveryPolicy<Error> recovery_policy_;
    std::optional<double> eta_scale_;
    std::size_t iteration_ = 0;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class GrippoLamparielloLucidiCondition
  {
   public:
    explicit GrippoLamparielloLucidiCondition(GrippoLamparielloLucidiParams params,
        RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{})
        : params_(params), recovery_policy_(std::move(recovery_policy))
    {
    }

    void begin_line_search(const double merit_0);

    [[nodiscard]] MeritConditionResult operator()(const double dmerit_da_0, const double merit_0,
        const MeritFunction<Error>& merit, const double alpha) const;

   private:
    GrippoLamparielloLucidiParams params_;
    RecoveryPolicy<Error> recovery_policy_;
    std::deque<double> merit_history_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class ZhangHagerNonmonotoneArmijoCondition
  {
   public:
    explicit ZhangHagerNonmonotoneArmijoCondition(ZhangHagerNonmonotoneParams params,
        RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{})
        : params_(params), recovery_policy_(std::move(recovery_policy))
    {
    }

    [[nodiscard]] MeritConditionResult operator()(const double dmerit_da_0, const double merit_0,
        const MeritFunction<Error>& merit, const double alpha);

   private:
    ZhangHagerNonmonotoneParams params_;
    RecoveryPolicy<Error> recovery_policy_;
    std::optional<double> reference_merit_;
    double reference_weight_ = 1.0;
  };

  template <typename Result>
  concept LineSearchConditionResult = requires(const Result& result) {
    { result.status } -> std::convertible_to<TrialStatus>;
    { result.recovery } -> std::convertible_to<std::optional<RecoveryAction>>;
  };

  template <typename Result>
  concept MeritLineSearchConditionResult =
      LineSearchConditionResult<Result> && requires(const Result& result) {
        { result.data.merit } -> std::convertible_to<double>;
      };

  template <typename Result>
  concept DMeritLineSearchConditionResult =
      MeritLineSearchConditionResult<Result> && requires(const Result& result) {
        { result.data.dmerit_da } -> std::convertible_to<double>;
      };

  template <typename Condition, typename Error>
  concept LineSearchCondition = requires(Condition& condition, const double dmerit_da_0,
      const double merit_0, const MeritFunction<Error>& merit, const double alpha) {
    { condition(dmerit_da_0, merit_0, merit, alpha) } -> LineSearchConditionResult;
  };

  template <typename Condition, typename Error>
  concept MeritLineSearchCondition =
      LineSearchCondition<Condition, Error> &&
      requires(Condition& condition, const double dmerit_da_0, const double merit_0,
          const MeritFunction<Error>& merit, const double alpha) {
        { condition(dmerit_da_0, merit_0, merit, alpha) } -> MeritLineSearchConditionResult;
      };

  template <typename Condition, typename Error>
  concept DMeritLineSearchCondition =
      MeritLineSearchCondition<Condition, Error> &&
      requires(Condition& condition, const double dmerit_da_0, const double merit_0,
          const MeritFunction<Error>& merit, const double alpha) {
        { condition(dmerit_da_0, merit_0, merit, alpha) } -> DMeritLineSearchConditionResult;
      };

  [[nodiscard]] inline std::optional<double> recovery_contraction_factor(
      const RecoveryAction& recovery, const double too_high_factor)
  {
    if (const auto* contraction = std::get_if<ContractStep>(&recovery))
    {
      if (std::isfinite(contraction->factor) && contraction->factor > 0.0 &&
          contraction->factor < 1.0)
        return contraction->factor;
      return std::nullopt;
    }
    if (std::holds_alternative<TreatAsTooHigh>(recovery)) return too_high_factor;
    return std::nullopt;
  }

  template <typename Error>
  struct MeritAndDerivativeResult
  {
    double value;
    double dmerit_da_alpha;
    std::optional<Error> error;
  };

  template <typename Error>
  [[nodiscard]] MeritAndDerivativeResult<Error> evaluate_merit_and_left_derivative(
      const MeritFunction<Error>& merit, const double alpha)
  {
    const MeritResult<Error> result_at_alpha = merit(alpha);
    if (result_at_alpha.error.has_value())
    {
      return {.value = 0.0, .dmerit_da_alpha = 0.0, .error = result_at_alpha.error};
    }

    const double finite_difference_step =
        std::min(std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(alpha)),
            0.5 * alpha);
    const MeritResult<Error> result_left = merit(alpha - finite_difference_step);
    if (result_left.error.has_value())
    {
      return {.value = 0.0, .dmerit_da_alpha = 0.0, .error = result_left.error};
    }

    return {.value = result_at_alpha.value,
        .dmerit_da_alpha = (result_at_alpha.value - result_left.value) / finite_difference_step,
        .error = std::nullopt};
  }

  template <typename Condition, typename Error>
    requires LineSearchCondition<Condition, Error>
  class Backtracking final : public LineSearch<Error>
  {
   public:
    explicit Backtracking(
        const StepControlParams& params, double reduction_factor, Condition condition)
        : LineSearch<Error>(params),
          reduction_factor_(reduction_factor),
          condition_(std::move(condition))
    {
    }

    [[nodiscard]] double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) override
    {
      if constexpr (requires { condition_.begin_line_search(merit_0); })
      {
        condition_.begin_line_search(merit_0);
      }

      double alpha = this->params_.alpha_init;
      for (int iter = 0; alpha >= minimum_alpha && iter < this->params_.max_iter; ++iter)
      {
        const auto result = condition_(dmerit_da_0, merit_0, merit, alpha);
        if (result.recovery)
        {
          const auto factor = recovery_contraction_factor(*result.recovery, reduction_factor_);
          if (!factor) return 0.0;
          alpha *= *factor;
          continue;
        }

        if (result.status == TrialStatus::accepted) return alpha;
        if (result.status == TrialStatus::too_low) return 0.0;

        alpha *= reduction_factor_;
      }

      return 0.0;
    }

   private:
    static constexpr double minimum_alpha = 1.0e-12;
    double reduction_factor_;
    Condition condition_;
  };

  /**
   * Backtrack using the minimizer of a parabola interpolating the merit at zero, at the current
   * trial and its derivative at zero. The interpolated trial is safeguarded to reduce the
   * current step by a factor between 0.1 and 0.5.
   */
  template <typename Condition, typename Error>
    requires MeritLineSearchCondition<Condition, Error>
  class SafeguardedQuadraticBacktracking final : public LineSearch<Error>
  {
   public:
    explicit SafeguardedQuadraticBacktracking(const StepControlParams& params, Condition condition)
        : LineSearch<Error>(params), condition_(std::move(condition))
    {
    }

    [[nodiscard]] double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) override
    {
      if constexpr (requires { condition_.begin_line_search(merit_0); })
      {
        condition_.begin_line_search(merit_0);
      }

      double alpha = this->params_.alpha_init;
      for (int iter = 0; alpha >= minimum_alpha && iter < this->params_.max_iter; ++iter)
      {
        const auto result = condition_(dmerit_da_0, merit_0, merit, alpha);
        if (result.recovery)
        {
          const auto factor =
              recovery_contraction_factor(*result.recovery, maximum_reduction_factor);
          if (!factor) return 0.0;
          alpha *= *factor;
          continue;
        }

        if (result.status == TrialStatus::accepted) return alpha;
        if (result.status == TrialStatus::too_low) return 0.0;

        double reduction_factor = maximum_reduction_factor;
        const double quadratic_denominator =
            2.0 * (result.data.merit - merit_0 - alpha * dmerit_da_0);
        if (std::isfinite(quadratic_denominator) && quadratic_denominator > 0.0)
        {
          const double quadratic_minimizer = -dmerit_da_0 * alpha * alpha / quadratic_denominator;
          if (std::isfinite(quadratic_minimizer))
          {
            reduction_factor = std::clamp(
                quadratic_minimizer / alpha, minimum_reduction_factor, maximum_reduction_factor);
          }
        }

        alpha *= reduction_factor;
      }

      return 0.0;
    }

   private:
    static constexpr double minimum_alpha = 1.0e-12;
    static constexpr double minimum_reduction_factor = 0.1;
    static constexpr double maximum_reduction_factor = 0.5;
    Condition condition_;
  };

  /**
   * Bracket an acceptable step by doubling a trial that is too low and bisecting once an upper
   * bound is known. For the weak Wolfe conditions, this is Algorithm 2.10 in Andrei, Modern
   * Numerical Nonlinear Optimization (2022)
   */
  template <typename Condition, typename Error>
    requires LineSearchCondition<Condition, Error>
  class SimpleBisection final : public LineSearch<Error>
  {
   public:
    explicit SimpleBisection(const StepControlParams& params, Condition condition)
        : LineSearch<Error>(params), condition_(std::move(condition))
    {
    }

    [[nodiscard]] double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) override
    {
      if constexpr (requires { condition_.begin_line_search(merit_0); })
      {
        condition_.begin_line_search(merit_0);
      }

      double lower_alpha = 0.0;
      std::optional<double> upper_alpha;
      double alpha = this->params_.alpha_init;
      for (int iter = 0; alpha >= minimum_alpha && iter < this->params_.max_iter; ++iter)
      {
        const auto condition_result = condition_(dmerit_da_0, merit_0, merit, alpha);
        if (!condition_result.recovery && condition_result.status == TrialStatus::accepted)
        {
          return alpha;
        }

        std::optional<double> recovery_factor;
        if (condition_result.recovery)
        {
          recovery_factor = recovery_contraction_factor(*condition_result.recovery, 0.5);
          if (!recovery_factor) return 0.0;
          upper_alpha = alpha;
        }
        else if (condition_result.status == TrialStatus::too_high)
          upper_alpha = alpha;
        else if (condition_result.status == TrialStatus::too_low)
          lower_alpha = alpha;
        else
          return 0.0;

        if (upper_alpha.has_value())
        {
          const double next_alpha =
              recovery_factor ? lower_alpha + *recovery_factor * (*upper_alpha - lower_alpha)
                              : std::midpoint(lower_alpha, *upper_alpha);
          if (next_alpha == lower_alpha || next_alpha == *upper_alpha) return 0.0;
          alpha = next_alpha;
        }
        else
        {
          alpha *= 2.0;
        }
      }

      return 0.0;
    }

   private:
    static constexpr double minimum_alpha = 1.0e-12;
    Condition condition_;
  };

  template <typename Condition, typename Error>
    requires DMeritLineSearchCondition<Condition, Error>
  class MoreThuente final : public LineSearch<Error>
  {
   public:
    explicit MoreThuente(
        const StepControlParams& params, double sufficient_decrease_c1, Condition condition)
        : LineSearch<Error>(params),
          sufficient_decrease_c1_(sufficient_decrease_c1),
          condition_(std::move(condition))
    {
    }

    [[nodiscard]] double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) override
    {
      if constexpr (requires { condition_.begin_line_search(merit_0); })
      {
        condition_.begin_line_search(merit_0);
      }

      if (!std::isfinite(dmerit_da_0) || dmerit_da_0 >= 0.0 || !std::isfinite(merit_0)) return 0.0;

      const double sufficient_decrease_slope = sufficient_decrease_c1_ * dmerit_da_0;
      Point best{.alpha = 0.0, .merit = merit_0, .derivative = dmerit_da_0};
      Point other = best;
      double alpha = std::clamp(this->params_.alpha_init, minimum_alpha, maximum_alpha);
      double interval_minimum = 0.0;
      double interval_maximum = alpha + extrapolation_upper * alpha;
      double width = maximum_alpha - minimum_alpha;
      double previous_width = 2.0 * width;
      bool bracketed = false;
      bool stage_one = true;

      for (int iter = 0; iter < this->params_.max_iter; ++iter)
      {
        const auto result = condition_(dmerit_da_0, merit_0, merit, alpha);
        if (result.recovery)
        {
          const auto factor = recovery_contraction_factor(*result.recovery, 0.5);
          if (!factor) return 0.0;
          const double next_alpha = best.alpha + *factor * (alpha - best.alpha);
          if (!std::isfinite(next_alpha) || next_alpha == best.alpha || next_alpha == alpha)
            return 0.0;
          alpha = next_alpha;
          continue;
        }
        if (!std::isfinite(result.data.merit) || !std::isfinite(result.data.dmerit_da)) return 0.0;

        Point trial{
            .alpha = alpha, .merit = result.data.merit, .derivative = result.data.dmerit_da};
        if (result.status == TrialStatus::accepted) return alpha;

        const double sufficient_decrease_limit = merit_0 + alpha * sufficient_decrease_slope;
        if (stage_one && trial.merit <= sufficient_decrease_limit && trial.derivative >= 0.0)
          stage_one = false;

        bool update_successful = false;
        if (stage_one && trial.merit <= best.merit && trial.merit > sufficient_decrease_limit)
        {
          Point modified_best = modified(best, merit_0, sufficient_decrease_slope);
          Point modified_other = modified(other, merit_0, sufficient_decrease_slope);
          Point modified_trial = modified(trial, merit_0, sufficient_decrease_slope);
          update_successful = update_step(modified_best, modified_other, modified_trial, bracketed,
              interval_minimum, interval_maximum, alpha);
          best = unmodified(modified_best, merit_0, sufficient_decrease_slope);
          other = unmodified(modified_other, merit_0, sufficient_decrease_slope);
        }
        else
        {
          update_successful =
              update_step(best, other, trial, bracketed, interval_minimum, interval_maximum, alpha);
        }
        if (!update_successful) return 0.0;

        if (bracketed)
        {
          if (std::abs(other.alpha - best.alpha) >= interval_contraction * previous_width)
            alpha = std::midpoint(best.alpha, other.alpha);
          previous_width = width;
          width = std::abs(other.alpha - best.alpha);
          interval_minimum = std::min(best.alpha, other.alpha);
          interval_maximum = std::max(best.alpha, other.alpha);
        }
        else
        {
          interval_minimum = alpha + extrapolation_lower * (alpha - best.alpha);
          interval_maximum = alpha + extrapolation_upper * (alpha - best.alpha);
          if (interval_minimum > interval_maximum) std::swap(interval_minimum, interval_maximum);
        }

        alpha = std::clamp(alpha, minimum_alpha, maximum_alpha);
        if (!std::isfinite(alpha) ||
            (bracketed &&
                (alpha <= interval_minimum || alpha >= interval_maximum ||
                    interval_maximum - interval_minimum <= interval_tolerance * interval_maximum)))
          return 0.0;
      }

      return 0.0;
    }

   private:
    struct Point
    {
      double alpha;
      double merit;
      double derivative;
    };

    [[nodiscard]] static Point modified(
        Point point, const double merit_0, const double sufficient_decrease_slope)
    {
      point.merit -= merit_0 + point.alpha * sufficient_decrease_slope;
      point.derivative -= sufficient_decrease_slope;
      return point;
    }

    [[nodiscard]] static Point unmodified(
        Point point, const double merit_0, const double sufficient_decrease_slope)
    {
      point.merit += merit_0 + point.alpha * sufficient_decrease_slope;
      point.derivative += sufficient_decrease_slope;
      return point;
    }

    [[nodiscard]] static double cubic_discriminant(
        const double theta, const double derivative_a, const double derivative_b)
    {
      const double scale =
          std::max({std::abs(theta), std::abs(derivative_a), std::abs(derivative_b)});
      if (scale == 0.0) return 0.0;
      const double scaled_theta = theta / scale;
      return scale * std::sqrt(std::max(0.0, scaled_theta * scaled_theta -
                                                 (derivative_a / scale) * (derivative_b / scale)));
    }

    struct ThetaGamma
    {
      double theta;
      double gamma;
    };

    [[nodiscard]] static ThetaGamma cubic_theta_gamma(
        const Point& p1, const Point& p2, const bool flip)
    {
      const double theta =
          3.0 * (p1.merit - p2.merit) / (p2.alpha - p1.alpha) + p1.derivative + p2.derivative;
      double gamma = cubic_discriminant(theta, p1.derivative, p2.derivative);
      if (flip) gamma = -gamma;
      return {.theta = theta, .gamma = gamma};
    }

    [[nodiscard]] static bool update_step(Point& best, Point& other, const Point& trial,
        bool& bracketed, const double minimum, const double maximum, double& next_alpha)
    {
      if (trial.alpha == best.alpha || minimum > maximum) return false;

      const double derivative_sign =
          std::copysign(1.0, trial.derivative) * std::copysign(1.0, best.derivative);
      double cubic_alpha = 0.0;
      double quadratic_alpha = 0.0;
      double candidate = 0.0;

      if (trial.merit > best.merit)
      {
        const auto [theta, gamma] = cubic_theta_gamma(best, trial, trial.alpha < best.alpha);
        const double p = gamma - best.derivative + theta;
        const double q = 2.0 * gamma - best.derivative + trial.derivative;
        const double quadratic_denominator =
            (best.merit - trial.merit) / (trial.alpha - best.alpha) + best.derivative;
        if (q == 0.0 || quadratic_denominator == 0.0) return false;
        cubic_alpha = best.alpha + (p / q) * (trial.alpha - best.alpha);
        quadratic_alpha =
            best.alpha + 0.5 * best.derivative / quadratic_denominator * (trial.alpha - best.alpha);
        candidate = std::abs(cubic_alpha - best.alpha) <= std::abs(quadratic_alpha - best.alpha)
                        ? cubic_alpha
                        : std::midpoint(cubic_alpha, quadratic_alpha);
        bracketed = true;
      }
      else if (derivative_sign < 0.0)
      {
        const auto [theta, gamma] = cubic_theta_gamma(best, trial, trial.alpha > best.alpha);
        const double p = gamma - trial.derivative + theta;
        const double q = 2.0 * gamma - trial.derivative + best.derivative;
        const double secant_denominator = trial.derivative - best.derivative;
        if (q == 0.0 || secant_denominator == 0.0) return false;
        cubic_alpha = trial.alpha + (p / q) * (best.alpha - trial.alpha);
        quadratic_alpha =
            trial.alpha + trial.derivative / secant_denominator * (best.alpha - trial.alpha);
        candidate = std::abs(cubic_alpha - trial.alpha) > std::abs(quadratic_alpha - trial.alpha)
                        ? cubic_alpha
                        : quadratic_alpha;
        bracketed = true;
      }
      else if (std::abs(trial.derivative) < std::abs(best.derivative))
      {
        const auto [theta, gamma] = cubic_theta_gamma(best, trial, trial.alpha > best.alpha);
        const double p = gamma - trial.derivative + theta;
        const double q = 2.0 * gamma + best.derivative - trial.derivative;
        const double ratio = q == 0.0 ? 0.0 : p / q;
        if (ratio < 0.0 && gamma != 0.0)
          cubic_alpha = trial.alpha + ratio * (best.alpha - trial.alpha);
        else
          cubic_alpha = trial.alpha > best.alpha ? maximum : minimum;

        const double secant_denominator = trial.derivative - best.derivative;
        if (secant_denominator == 0.0) return false;
        quadratic_alpha =
            trial.alpha + trial.derivative / secant_denominator * (best.alpha - trial.alpha);
        if (bracketed)
        {
          candidate = std::abs(cubic_alpha - trial.alpha) < std::abs(quadratic_alpha - trial.alpha)
                          ? cubic_alpha
                          : quadratic_alpha;
          const double safeguarded =
              trial.alpha + interval_contraction * (other.alpha - trial.alpha);
          candidate = trial.alpha > best.alpha ? std::min(candidate, safeguarded)
                                               : std::max(candidate, safeguarded);
        }
        else
        {
          candidate = std::abs(cubic_alpha - trial.alpha) > std::abs(quadratic_alpha - trial.alpha)
                          ? cubic_alpha
                          : quadratic_alpha;
          candidate = std::clamp(candidate, minimum, maximum);
        }
      }
      else if (bracketed)
      {
        if (other.alpha == trial.alpha) return false;
        const auto [theta, gamma] = cubic_theta_gamma(trial, other, trial.alpha > other.alpha);
        const double p = gamma - trial.derivative + theta;
        const double q = 2.0 * gamma - trial.derivative + other.derivative;
        if (q == 0.0) return false;
        candidate = trial.alpha + (p / q) * (other.alpha - trial.alpha);
      }
      else
      {
        candidate = trial.alpha > best.alpha ? maximum : minimum;
      }

      if (!std::isfinite(candidate)) return false;

      if (trial.merit > best.merit)
      {
        other = trial;
      }
      else
      {
        if (derivative_sign < 0.0) other = best;
        best = trial;
      }
      next_alpha = candidate;
      return true;
    }

    static constexpr double minimum_alpha = 1.0e-12;
    static constexpr double maximum_alpha = 1.0e12;
    static constexpr double interval_tolerance = 10.0 * std::numeric_limits<double>::epsilon();
    static constexpr double interval_contraction = 0.66;
    static constexpr double extrapolation_lower = 1.1;
    static constexpr double extrapolation_upper = 4.0;
    double sufficient_decrease_c1_;
    Condition condition_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class HagerZhang final : public LineSearch<Error>
  {
   public:
    explicit HagerZhang(const StepControlParams& params, HagerZhangParams hager_zhang_params,
        RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{});

    [[nodiscard]] double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) override;

   private:
    struct Point
    {
      double alpha;
      double merit;
      double derivative;
    };

    struct Interval
    {
      Point a;
      Point b;
    };

    struct UpdateResult
    {
      Interval interval;
      bool successful;
    };

    struct EvaluationContext
    {
      double dmerit_da_0;
      double merit_0;
      double epsilon_k;
      const MeritFunction<Error>& merit;
    };

    [[nodiscard]] bool evaluate(double alpha, const double lower_alpha, Point& point,
        const EvaluationContext& context) const;
    [[nodiscard]] bool satisfies_line_search_conditions(
        const Point& point, const EvaluationContext& context) const;
    [[nodiscard]] UpdateResult update(
        const Interval& interval, const Point& point, const EvaluationContext& context) const;
    [[nodiscard]] UpdateResult secant2(
        const Interval& interval, const EvaluationContext& context) const;
    [[nodiscard]] double secant(const Point& a, const Point& b) const;

    HagerZhangParams hager_zhang_params_;
    RecoveryPolicy<Error> recovery_policy_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  MeritConditionResult ArmijoCondition<Error>::operator()(const double dmerit_da_0,
      const double merit_0, const MeritFunction<Error>& merit, const double alpha) const
  {
    if (!std::isfinite(alpha) || alpha <= 0.0)
      return invalid_alpha_result<MeritConditionData>(alpha);

    const MeritResult<Error> merit_alpha = merit(alpha);
    if (merit_alpha.error.has_value())
    {
      return {.status = TrialStatus::too_high,
          .recovery = recovery_policy_(*merit_alpha.error),
          .data = {.merit = merit_alpha.value}};
    }

    const bool accepted = merit_alpha.value <= merit_0 + params_.c1 * alpha * dmerit_da_0;
    return {.status = accepted ? TrialStatus::accepted : TrialStatus::too_high,
        .recovery = std::nullopt,
        .data = {.merit = merit_alpha.value}};
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  MeritConditionResult GoldsteinCondition<Error>::operator()(const double dmerit_da_0,
      const double merit_0, const MeritFunction<Error>& merit, const double alpha) const
  {
    if (!std::isfinite(alpha) || alpha <= 0.0)
      return invalid_alpha_result<MeritConditionData>(alpha);

    const MeritResult<Error> merit_alpha = merit(alpha);
    if (merit_alpha.error.has_value())
    {
      return {.status = TrialStatus::too_high,
          .recovery = recovery_policy_(*merit_alpha.error),
          .data = {.merit = merit_alpha.value}};
    }

    const double upper_bound = merit_0 + params_.rho * alpha * dmerit_da_0;
    const double lower_bound = merit_0 + (1.0 - params_.rho) * alpha * dmerit_da_0;

    const bool accepted = merit_alpha.value <= upper_bound && merit_alpha.value >= lower_bound;
    TrialStatus status = TrialStatus::too_low;
    if (accepted)
      status = TrialStatus::accepted;
    else if (merit_alpha.value > upper_bound)
      status = TrialStatus::too_high;
    return {.status = status, .recovery = std::nullopt, .data = {.merit = merit_alpha.value}};
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  DMeritConditionResult WeakWolfeCondition<Error>::operator()(const double dmerit_da_0,
      const double merit_0, const MeritFunction<Error>& merit, const double alpha) const
  {
    if (!std::isfinite(alpha) || alpha <= 0.0)
      return invalid_alpha_result<DMeritConditionData>(alpha);

    const MeritAndDerivativeResult<Error> merit_alpha =
        evaluate_merit_and_left_derivative(merit, alpha);
    if (merit_alpha.error.has_value())
    {
      return {.status = TrialStatus::too_high,
          .recovery = recovery_policy_(*merit_alpha.error),
          .data = {.merit = merit_alpha.value, .dmerit_da = merit_alpha.dmerit_da_alpha}};
    }

    const bool sufficient_decrease =
        merit_alpha.value <= merit_0 + params_.c1 * alpha * dmerit_da_0;
    const bool curvature = merit_alpha.dmerit_da_alpha >= params_.c2 * dmerit_da_0;

    const bool accepted = sufficient_decrease && curvature;
    TrialStatus status = TrialStatus::too_high;
    if (accepted)
      status = TrialStatus::accepted;
    else if (sufficient_decrease)
      status = TrialStatus::too_low;
    return {.status = status,
        .recovery = std::nullopt,
        .data = {.merit = merit_alpha.value, .dmerit_da = merit_alpha.dmerit_da_alpha}};
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  DMeritConditionResult StrongWolfeCondition<Error>::operator()(const double dmerit_da_0,
      const double merit_0, const MeritFunction<Error>& merit, const double alpha) const
  {
    if (!std::isfinite(alpha) || alpha <= 0.0)
      return invalid_alpha_result<DMeritConditionData>(alpha);

    const MeritAndDerivativeResult<Error> merit_alpha =
        evaluate_merit_and_left_derivative(merit, alpha);
    if (merit_alpha.error.has_value())
    {
      return {.status = TrialStatus::too_high,
          .recovery = recovery_policy_(*merit_alpha.error),
          .data = {.merit = merit_alpha.value, .dmerit_da = merit_alpha.dmerit_da_alpha}};
    }

    const bool sufficient_decrease =
        merit_alpha.value <= merit_0 + params_.c1 * alpha * dmerit_da_0;
    const bool strong_curvature =
        std::abs(merit_alpha.dmerit_da_alpha) <= params_.c2 * std::abs(dmerit_da_0);

    const bool accepted = sufficient_decrease && strong_curvature;
    const bool derivative_is_too_positive =
        merit_alpha.dmerit_da_alpha > params_.c2 * std::abs(dmerit_da_0);
    TrialStatus status = TrialStatus::too_low;
    if (accepted)
      status = TrialStatus::accepted;
    else if (!sufficient_decrease || derivative_is_too_positive)
      status = TrialStatus::too_high;
    return {.status = status,
        .recovery = std::nullopt,
        .data = {.merit = merit_alpha.value, .dmerit_da = merit_alpha.dmerit_da_alpha}};
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  DMeritConditionResult DaiKouCondition<Error>::operator()(const double dmerit_da_0,
      const double merit_0, const MeritFunction<Error>& merit, const double alpha)
  {
    if (!eta_scale_.has_value()) eta_scale_ = params_.eta_relative * std::abs(dmerit_da_0);

    if (!std::isfinite(alpha) || alpha <= 0.0)
      return invalid_alpha_result<DMeritConditionData>(alpha);

    const MeritAndDerivativeResult<Error> merit_dmerit_alpha =
        evaluate_merit_and_left_derivative(merit, alpha);
    if (merit_dmerit_alpha.error.has_value())
    {
      return {.status = TrialStatus::too_high,
          .recovery = recovery_policy_(*merit_dmerit_alpha.error),
          .data = {
              .merit = merit_dmerit_alpha.value, .dmerit_da = merit_dmerit_alpha.dmerit_da_alpha}};
    }

    const double sequence_index = static_cast<double>(iteration_ + 1);
    const double eta_k = *eta_scale_ / (sequence_index * sequence_index);
    const double relaxed_decrease = std::min(
        params_.epsilon * std::abs(dmerit_da_0), params_.rho * alpha * dmerit_da_0 + eta_k);
    const bool sufficient_decrease = merit_dmerit_alpha.value <= merit_0 + relaxed_decrease;
    const bool curvature = merit_dmerit_alpha.dmerit_da_alpha >= params_.sigma * dmerit_da_0;

    const bool accepted = sufficient_decrease && curvature;
    TrialStatus status = TrialStatus::too_high;
    if (accepted)
    {
      status = TrialStatus::accepted;
      ++iteration_;
    }
    else if (sufficient_decrease)
      status = TrialStatus::too_low;
    return {.status = status,
        .recovery = std::nullopt,
        .data = {
            .merit = merit_dmerit_alpha.value, .dmerit_da = merit_dmerit_alpha.dmerit_da_alpha}};
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  void GrippoLamparielloLucidiCondition<Error>::begin_line_search(const double merit_0)
  {
    merit_history_.push_back(merit_0);

    const std::size_t maximum_number_of_values = static_cast<std::size_t>(params_.max_history) + 1;
    while (merit_history_.size() > maximum_number_of_values)
    {
      merit_history_.pop_front();
    }
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  MeritConditionResult GrippoLamparielloLucidiCondition<Error>::operator()(const double dmerit_da_0,
      const double merit_0, const MeritFunction<Error>& merit, const double alpha) const
  {
    if (!std::isfinite(alpha) || alpha <= 0.0)
      return invalid_alpha_result<MeritConditionData>(alpha);

    const MeritResult<Error> merit_alpha = merit(alpha);
    if (merit_alpha.error.has_value())
    {
      return {.status = TrialStatus::too_high,
          .recovery = recovery_policy_(*merit_alpha.error),
          .data = {.merit = merit_alpha.value}};
    }

    const double maximum_previous_merit =
        merit_history_.empty() ? merit_0
                               : *std::max_element(merit_history_.begin(), merit_history_.end());

    const bool accepted =
        merit_alpha.value <= maximum_previous_merit + params_.rho * alpha * dmerit_da_0;
    return {.status = accepted ? TrialStatus::accepted : TrialStatus::too_high,
        .recovery = std::nullopt,
        .data = {.merit = merit_alpha.value}};
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  MeritConditionResult ZhangHagerNonmonotoneArmijoCondition<Error>::operator()(
      const double dmerit_da_0, const double merit_0, const MeritFunction<Error>& merit,
      const double alpha)
  {
    if (!reference_merit_.has_value()) reference_merit_ = merit_0;

    if (!std::isfinite(alpha) || alpha <= 0.0)
      return invalid_alpha_result<MeritConditionData>(alpha);

    const MeritResult<Error> merit_alpha = merit(alpha);
    if (merit_alpha.error.has_value())
    {
      return {.status = TrialStatus::too_high,
          .recovery = recovery_policy_(*merit_alpha.error),
          .data = {.merit = merit_alpha.value}};
    }

    const bool accepted =
        merit_alpha.value <= *reference_merit_ + params_.rho * alpha * dmerit_da_0;
    if (accepted)
    {
      const double next_reference_weight = params_.eta * reference_weight_ + 1.0;
      reference_merit_ = (params_.eta * reference_weight_ * *reference_merit_ + merit_alpha.value) /
                         next_reference_weight;
      reference_weight_ = next_reference_weight;
    }

    return {.status = accepted ? TrialStatus::accepted : TrialStatus::too_high,
        .recovery = std::nullopt,
        .data = {.merit = merit_alpha.value}};
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  HagerZhang<Error>::HagerZhang(const StepControlParams& params,
      HagerZhangParams hager_zhang_params, RecoveryPolicy<Error> recovery_policy)
      : LineSearch<Error>(params),
        hager_zhang_params_(hager_zhang_params),
        recovery_policy_(std::move(recovery_policy))
  {
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  bool HagerZhang<Error>::evaluate(
      double alpha, const double lower_alpha, Point& point, const EvaluationContext& context) const
  {
    for (int iter = 0; iter < this->params_.max_iter; ++iter)
    {
      if (!std::isfinite(alpha) || alpha < lower_alpha) return false;
      if (alpha == 0.0)
      {
        point = {.alpha = 0.0, .merit = context.merit_0, .derivative = context.dmerit_da_0};
        return true;
      }

      const MeritAndDerivativeResult<Error> result =
          evaluate_merit_and_left_derivative(context.merit, alpha);
      if (result.error)
      {
        const auto factor =
            recovery_contraction_factor(recovery_policy_(*result.error), hager_zhang_params_.theta);
        if (!factor) return false;
        const double next_alpha = lower_alpha + *factor * (alpha - lower_alpha);
        if (next_alpha == lower_alpha || next_alpha == alpha) return false;
        alpha = next_alpha;
        continue;
      }
      if (!std::isfinite(result.value) || !std::isfinite(result.dmerit_da_alpha)) return false;

      point = {.alpha = alpha, .merit = result.value, .derivative = result.dmerit_da_alpha};
      return true;
    }
    return false;
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  bool HagerZhang<Error>::satisfies_line_search_conditions(
      const Point& point, const EvaluationContext& context) const
  {
    const auto& hz = hager_zhang_params_;
    const bool curvature = point.derivative >= hz.sigma * context.dmerit_da_0;
    const bool ls1 =
        curvature && point.merit <= context.merit_0 + hz.delta * point.alpha * context.dmerit_da_0;
    const bool ls2 = curvature &&
                     point.derivative <= (2.0 * hz.delta - 1.0) * context.dmerit_da_0 &&
                     point.merit <= context.merit_0 + context.epsilon_k;
    return ls1 || ls2;
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  typename HagerZhang<Error>::UpdateResult HagerZhang<Error>::update(
      const Interval& interval, const Point& point, const EvaluationContext& context) const
  {
    if (!(point.alpha > interval.a.alpha && point.alpha < interval.b.alpha))
    {
      return {.interval = interval, .successful = true};
    }
    if (point.derivative >= 0.0)
    {
      return {.interval = {.a = interval.a, .b = point}, .successful = true};
    }
    if (point.merit <= context.merit_0 + context.epsilon_k)
    {
      return {.interval = {.a = point, .b = interval.b}, .successful = true};
    }

    Point lower = interval.a;
    Point upper = point;
    for (int iter = 0; iter < this->params_.max_iter; ++iter)
    {
      const double alpha =
          (1.0 - hager_zhang_params_.theta) * lower.alpha + hager_zhang_params_.theta * upper.alpha;
      if (!(alpha > lower.alpha && alpha < upper.alpha)) break;

      Point trial{};
      if (!evaluate(alpha, lower.alpha, trial, context))
        return {.interval = interval, .successful = false};
      if (trial.derivative >= 0.0)
      {
        return {.interval = {.a = lower, .b = trial}, .successful = true};
      }
      if (trial.merit <= context.merit_0 + context.epsilon_k)
        lower = trial;
      else
        upper = trial;
    }
    return {.interval = interval, .successful = false};
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  double HagerZhang<Error>::secant(const Point& a, const Point& b) const
  {
    const double denominator = b.derivative - a.derivative;
    if (denominator == 0.0 || !std::isfinite(denominator)) return 0.5 * (a.alpha + b.alpha);

    const double alpha = (a.alpha * b.derivative - b.alpha * a.derivative) / denominator;
    return std::isfinite(alpha) ? alpha : 0.5 * (a.alpha + b.alpha);
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  typename HagerZhang<Error>::UpdateResult HagerZhang<Error>::secant2(
      const Interval& interval, const EvaluationContext& context) const
  {
    Point first{};
    if (!evaluate(secant(interval.a, interval.b), interval.a.alpha, first, context))
      return {.interval = interval, .successful = false};

    UpdateResult result = update(interval, first, context);
    if (!result.successful) return result;

    double second_alpha = 0.0;
    bool use_second_secant = false;
    if (first.alpha == result.interval.b.alpha)
    {
      second_alpha = secant(interval.b, result.interval.b);
      use_second_secant = true;
    }
    else if (first.alpha == result.interval.a.alpha)
    {
      second_alpha = secant(interval.a, result.interval.a);
      use_second_secant = true;
    }
    if (!use_second_secant ||
        !(second_alpha > result.interval.a.alpha && second_alpha < result.interval.b.alpha))
    {
      return result;
    }

    Point second{};
    if (!evaluate(second_alpha, result.interval.a.alpha, second, context))
      return {.interval = interval, .successful = false};
    return update(result.interval, second, context);
  }

  template <typename Error>
    requires std::is_enum_v<Error>
  double HagerZhang<Error>::operator()(
      const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit)
  {
    const EvaluationContext context{.dmerit_da_0 = dmerit_da_0,
        .merit_0 = merit_0,
        .epsilon_k = hager_zhang_params_.epsilon * std::abs(merit_0),
        .merit = merit};

    if (!std::isfinite(dmerit_da_0) || dmerit_da_0 >= 0.0 || !std::isfinite(merit_0)) return 0.0;

    Point a{.alpha = 0.0, .merit = merit_0, .derivative = dmerit_da_0};
    Point b{};
    if (!evaluate(this->params_.alpha_init, 0.0, b, context)) return 0.0;
    if (satisfies_line_search_conditions(b, context)) return b.alpha;

    for (int iter = 0; b.derivative < 0.0 && iter < this->params_.max_iter; ++iter)
    {
      if (b.merit <= context.merit_0 + context.epsilon_k)
      {
        a = b;
        const double expanded_alpha = b.alpha * hager_zhang_params_.rho;
        Point expanded{};
        if (!evaluate(expanded_alpha, a.alpha, expanded, context)) return 0.0;
        b = expanded;
        if (satisfies_line_search_conditions(b, context)) return b.alpha;
        continue;
      }

      const double alpha =
          (1.0 - hager_zhang_params_.theta) * a.alpha + hager_zhang_params_.theta * b.alpha;
      if (!(alpha > a.alpha && alpha < b.alpha)) return 0.0;
      Point trial{};
      if (!evaluate(alpha, a.alpha, trial, context)) return 0.0;
      if (satisfies_line_search_conditions(trial, context)) return trial.alpha;
      if (trial.derivative >= 0.0)
        b = trial;
      else if (trial.merit <= context.merit_0 + context.epsilon_k)
        a = trial;
      else
        b = trial;
    }
    if (b.derivative < 0.0) return 0.0;

    Interval interval{.a = a, .b = b};
    for (int iter = 0; iter < this->params_.max_iter; ++iter)
    {
      if (satisfies_line_search_conditions(interval.a, context)) return interval.a.alpha;

      const double old_width = interval.b.alpha - interval.a.alpha;
      UpdateResult result = secant2(interval, context);
      if (!result.successful) return 0.0;
      if (result.interval.b.alpha - result.interval.a.alpha > hager_zhang_params_.gamma * old_width)
      {
        Point midpoint{};
        if (!evaluate(0.5 * (result.interval.a.alpha + result.interval.b.alpha),
                result.interval.a.alpha, midpoint, context))
          return 0.0;
        if (satisfies_line_search_conditions(midpoint, context)) return midpoint.alpha;
        result = update(result.interval, midpoint, context);
        if (!result.successful) return 0.0;
      }
      interval = result.interval;
    }
    return 0.0;
  }
}  // namespace Core::Utils::LineSearch

FOUR_C_NAMESPACE_CLOSE

#endif
