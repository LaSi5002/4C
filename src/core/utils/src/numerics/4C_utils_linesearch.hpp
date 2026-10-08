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

  template <typename Error>
    requires std::is_enum_v<Error>
  class AlwaysAbort
  {
   public:
    [[nodiscard]] RecoveryAction operator()(const Error) const { return AbortLineSearch{}; }
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class AlwaysTreatAsTooHigh
  {
   public:
    [[nodiscard]] RecoveryAction operator()(const Error) const { return TreatAsTooHigh{}; }
  };

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

  using MeritConditionResult = ConditionResult<MeritConditionData>;

  template <typename Error>
  using MeritFunction = std::function<MeritResult<Error>(double)>;

  template <typename Data>
  [[nodiscard]] ConditionResult<Data> invalid_alpha_result(const double alpha)
  {
    return {.status = std::isfinite(alpha) ? TrialStatus::too_low : TrialStatus::too_high,
        .recovery = std::nullopt,
        .data = Data{}};
  }

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

    virtual void reset() {}

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
  class GrippoLamparielloLucidiCondition
  {
   public:
    explicit GrippoLamparielloLucidiCondition(GrippoLamparielloLucidiParams params,
        RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{})
        : params_(params), recovery_policy_(std::move(recovery_policy))
    {
    }

    void begin_line_search(const double merit_0);

    void reset() { merit_history_.clear(); }

    [[nodiscard]] MeritConditionResult operator()(const double dmerit_da_0, const double merit_0,
        const MeritFunction<Error>& merit, const double alpha) const;

   private:
    GrippoLamparielloLucidiParams params_;
    RecoveryPolicy<Error> recovery_policy_;
    std::deque<double> merit_history_;
  };

  template <typename Result>
  concept LineSearchConditionResult = requires(const Result& result) {
    { result.status } -> std::convertible_to<TrialStatus>;
    { result.recovery } -> std::convertible_to<std::optional<RecoveryAction>>;
  };

  template <typename Condition, typename Error>
  concept LineSearchCondition = requires(Condition& condition, const double dmerit_da_0,
      const double merit_0, const MeritFunction<Error>& merit, const double alpha) {
    { condition(dmerit_da_0, merit_0, merit, alpha) } -> LineSearchConditionResult;
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

  class BacktrackingTrialBudget
  {
   public:
    explicit BacktrackingTrialBudget(const int max_trials) : max_trials_(max_trials) {}

    void record(const double alpha, const bool evaluable)
    {
      if (!evaluable) return;
      ++num_evaluable_trials_;
      last_evaluable_alpha_ = alpha;
    }

    [[nodiscard]] std::optional<double> exhausted_result() const
    {
      if (max_trials_ <= 0 || num_evaluable_trials_ < max_trials_) return std::nullopt;
      return last_evaluable_alpha_;
    }

   private:
    int max_trials_;
    int num_evaluable_trials_ = 0;
    std::optional<double> last_evaluable_alpha_;
  };

  template <typename Condition, typename Error>
    requires LineSearchCondition<Condition, Error>
  class Backtracking final : public LineSearch<Error>
  {
   public:
    explicit Backtracking(const StepControlParams& params, double reduction_factor,
        Condition condition, int max_trials = 0)
        : LineSearch<Error>(params),
          reduction_factor_(reduction_factor),
          condition_(std::move(condition)),
          max_trials_(max_trials)
    {
    }

    [[nodiscard]] double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) override
    {
      if constexpr (requires { condition_.begin_line_search(merit_0); })
      {
        condition_.begin_line_search(merit_0);
      }

      BacktrackingTrialBudget budget(max_trials_);
      double alpha = this->params_.alpha_init;
      for (int iter = 0; alpha >= minimum_alpha && iter < this->params_.max_iter; ++iter)
      {
        const auto result = condition_(dmerit_da_0, merit_0, merit, alpha);
        budget.record(alpha, !result.recovery.has_value());
        if (result.recovery)
        {
          const auto factor = recovery_contraction_factor(*result.recovery, reduction_factor_);
          if (!factor) return 0.0;
          alpha *= *factor;
        }
        else
        {
          if (result.status == TrialStatus::accepted) return alpha;
          if (result.status == TrialStatus::too_low) return 0.0;

          alpha *= reduction_factor_;
        }

        if (const auto alpha_on_exhaustion = budget.exhausted_result()) return *alpha_on_exhaustion;
      }

      return 0.0;
    }

    void reset() override
    {
      if constexpr (requires { condition_.reset(); }) condition_.reset();
    }

   private:
    static constexpr double minimum_alpha = 1.0e-12;
    double reduction_factor_;
    Condition condition_;
    int max_trials_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class ErrorAdaptiveNewton final : public LineSearch<Error>
  {
   public:
    explicit ErrorAdaptiveNewton(const StepControlParams& params, double reduction_factor)
        : LineSearch<Error>(params), reduction_factor_(reduction_factor)
    {
    }

    [[nodiscard]] double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) override
    {
      double alpha = this->params_.alpha_init;
      for (int iter = 0; alpha >= minimum_alpha && iter < this->params_.max_iter; ++iter)
      {
        const MeritResult<Error> result = merit(alpha);
        if (!result.error.has_value()) return alpha;

        alpha *= reduction_factor_;
      }

      return 0.0;
    }

   private:
    static constexpr double minimum_alpha = 1.0e-12;
    double reduction_factor_;
  };

  template <typename Error>
    requires std::is_enum_v<Error>
  class GoldenSectionSearch final : public LineSearch<Error>
  {
   public:
    explicit GoldenSectionSearch(const StepControlParams& params,
        RecoveryPolicy<Error> recovery_policy = RecoveryPolicy<Error>{}, double min_alpha = 0.0,
        double interval_tol = 1.0e-8)
        : LineSearch<Error>(params),
          recovery_policy_(std::move(recovery_policy)),
          min_alpha_(min_alpha),
          interval_tol_(interval_tol)
    {
    }

    [[nodiscard]] double operator()(
        const double dmerit_da_0, const double merit_0, MeritFunction<Error> merit) override
    {
      int iter = 0;
      const auto eval = [this, &merit, &iter](const double alpha) -> std::optional<double>
      {
        ++iter;
        const MeritResult<Error> result = merit(alpha);
        if (!result.error.has_value()) return result.value;

        const RecoveryAction recovery = recovery_policy_(*result.error);
        if (std::holds_alternative<AbortLineSearch>(recovery)) return std::nullopt;
        return std::numeric_limits<double>::infinity();
      };


      const auto apply_floor = [this, &eval](const double natural_alpha) -> double
      {
        if (min_alpha_ <= natural_alpha) return natural_alpha;
        const auto min_alpha_result = eval(min_alpha_);
        if (!min_alpha_result || !std::isfinite(*min_alpha_result)) return natural_alpha;
        return min_alpha_;
      };

      if (!std::isfinite(dmerit_da_0) || dmerit_da_0 >= 0.0 || !std::isfinite(merit_0))
        return apply_floor(0.0);

      double lo = 0.0, merit_lo = merit_0;
      double mid = this->params_.alpha_init;
      auto merit_mid_opt = eval(mid);

      while (merit_mid_opt && !std::isfinite(*merit_mid_opt) && iter < this->params_.max_iter &&
             mid * error_contraction_factor >= minimum_alpha)
      {
        mid *= error_contraction_factor;
        merit_mid_opt = eval(mid);
      }

      if (!merit_mid_opt) return apply_floor(0.0);
      double merit_mid = *merit_mid_opt;
      double hi = mid, merit_hi = merit_mid;

      while (merit_mid < merit_lo)
      {
        if (iter >= this->params_.max_iter) return apply_floor(0.0);
        hi = mid * growth_factor;
        const auto merit_hi_opt = eval(hi);
        if (!merit_hi_opt) return apply_floor(0.0);
        merit_hi = *merit_hi_opt;
        if (merit_hi >= merit_mid) break;

        lo = mid;
        merit_lo = merit_mid;
        mid = hi;
        merit_mid = merit_hi;
      }

      double a = lo, b = hi;
      double x1 = a + invphi2 * (b - a);
      double x2 = a + invphi * (b - a);
      auto f1_opt = eval(x1);
      if (!f1_opt) return apply_floor(0.0);
      double f1 = *f1_opt;
      auto f2_opt = eval(x2);
      if (!f2_opt) return apply_floor(0.0);
      double f2 = *f2_opt;

      while (iter < this->params_.max_iter && (b - a) > interval_tol_ * (1.0 + b))
      {
        if (f1 < f2)
        {
          b = x2;
          x2 = x1;
          f2 = f1;
          x1 = a + invphi2 * (b - a);
          const auto f1_opt2 = eval(x1);
          if (!f1_opt2) return apply_floor(0.0);
          f1 = *f1_opt2;
        }
        else
        {
          a = x1;
          x1 = x2;
          f1 = f2;
          x2 = a + invphi * (b - a);
          const auto f2_opt2 = eval(x2);
          if (!f2_opt2) return apply_floor(0.0);
          f2 = *f2_opt2;
        }
      }

      const double best_alpha = (f1 <= f2) ? x1 : x2;
      const double best_merit = std::min(f1, f2);
      if (!std::isfinite(best_merit)) return apply_floor(0.0);
      if (best_merit >= merit_0 || best_alpha <= 0.0) return apply_floor(0.0);
      return apply_floor(best_alpha);
    }

   private:
    static constexpr double growth_factor = 2.0;
    static constexpr double error_contraction_factor = 0.5;
    static constexpr double minimum_alpha = 1.0e-12;
    static constexpr double invphi = 0.6180339887498949;
    static constexpr double invphi2 = 0.3819660112501051;
    RecoveryPolicy<Error> recovery_policy_;
    double min_alpha_;
    double interval_tol_;
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

}  // namespace Core::Utils::LineSearch

FOUR_C_NAMESPACE_CLOSE

#endif
