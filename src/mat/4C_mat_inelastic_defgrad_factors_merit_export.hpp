// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_MERIT_EXPORT_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_MERIT_EXPORT_HPP

#include "4C_config.hpp"

#include "4C_mat_inelastic_defgrad_factors.hpp"
#include "4C_utils_enum.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace MeritExportDebug
{
  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;
  namespace LocalNewtonLineSearch = Core::Utils::LineSearch;

  //! Debug-only export of local-Newton merit-function curves phi(alpha), used to inspect how the
  //! merit function looks at different local-Newton iterations (e.g. for thesis figures). Enabled
  //! by setting the FOURC_MERIT_CURVE_EXPORT_DIR environment variable to an output directory; a
  //! single cached bool check keeps this a no-op otherwise, so it never touches production runs.
  //!
  //! Shared (header-only, one true singleton via the inline-function local-static trick) between
  //! the line-search-globalized local Newton loop and the plain, unsafeguarded one, so both paths
  //! draw from the same trajectory-id counter and export budget.
  //!
  //! Export budget decays per local-Newton iteration index (32, 16, 8, 4, 2, 1, then 1 apiece for
  //! every deeper iteration up to kMaxTrackedIter), so early, harder iterations are
  //! over-represented while still guaranteeing at least one exported curve at every depth a
  //! trajectory reaches -- enough to trace a single hard trajectory's shape all the way to
  //! convergence. The budget is per-process: since the testing framework spawns a fresh 4C process
  //! per timestep/repetition, point this at a single targeted run, not the batch driver, if you
  //! want a genuinely global cap.
  //!
  //! Every export also carries a trajectory_id, assigned once per call to
  //! globalized_local_newton_with_line_search or local_newton_loop (i.e. once per Gauss point per
  //! global-Newton iteration), so exported curves from the same local-Newton solve can be grouped
  //! and sorted by iteration index after the fact.
  class MeritCurveExporter
  {
   public:
    static constexpr unsigned int kMaxTrackedIter = 150;

    static MeritCurveExporter& instance()
    {
      static MeritCurveExporter exporter;
      return exporter;
    }

    MeritCurveExporter(const MeritCurveExporter&) = delete;
    MeritCurveExporter& operator=(const MeritCurveExporter&) = delete;

    [[nodiscard]] int next_trajectory_id() { return trajectory_counter_.fetch_add(1); }

    template <typename Merit>
    void maybe_export(const Merit& merit, int trajectory_id, unsigned int local_newton_iter,
        double accepted_alpha)
    {
      if (!output_dir_.has_value()) return;
      if (local_newton_iter >= budgets_.size()) return;
      if (budgets_[local_newton_iter].fetch_sub(1, std::memory_order_relaxed) <= 0) return;

      write_csv(merit, trajectory_id, local_newton_iter, accepted_alpha);
    }

   private:
    MeritCurveExporter()
    {
      if (const char* dir = std::getenv("FOURC_MERIT_CURVE_EXPORT_DIR"))
      {
        output_dir_ = std::filesystem::path(dir);
        std::filesystem::create_directories(*output_dir_);
      }
      // Generous and mostly flat: this budget is shared across every trajectory (Gauss point x
      // global-Newton-iteration) in the whole process, and a multi-element mesh can easily have
      // dozens to hundreds of those competing for the same early-iteration slots. Debug-only
      // tooling, so the extra CSV files this can produce cost nothing that matters.
      static constexpr std::array<int, 10> initial_decay{
          300, 300, 300, 200, 200, 100, 100, 50, 50, 20};
      for (unsigned int i = 0; i < kMaxTrackedIter; ++i)
        budgets_[i].store(i < initial_decay.size() ? initial_decay[i] : 10);
    }

    // Golden-section search for the true minimizer of merit(alpha) on [lo, hi], treating any
    // alpha whose evaluation errors out as worse than every valid value. Assumes [lo, hi] is
    // already a tight bracket around the minimum -- when the valid region is a small island
    // within a much larger domain (e.g. right after a badly-overshot iteration, where only a
    // sliver near alpha=0 is valid before a failure region swallows the rest), golden section's
    // very first pair of probe points can land entirely outside that island and never recover,
    // since every subsequent comparison is an uninformative "infinity vs. infinity" tie. Callers
    // should seed [lo, hi] from a coarse scan (see write_csv) rather than passing the full range.
    template <typename Merit>
    [[nodiscard]] std::pair<double, LocalNewtonLineSearch::MeritResult<ViscoplastUtils::ErrorType>>
    golden_section_minimize(const Merit& merit, double lo, double hi)
    {
      constexpr double inv_phi = 0.6180339887498949;  // (sqrt(5)-1)/2
      constexpr int iterations = 100;

      const auto value_or_inf =
          [](const LocalNewtonLineSearch::MeritResult<ViscoplastUtils::ErrorType>& result)
      { return result.error.has_value() ? std::numeric_limits<double>::infinity() : result.value; };

      double a = lo, b = hi;
      double c = b - inv_phi * (b - a);
      double d = a + inv_phi * (b - a);
      auto fc = merit(c);
      auto fd = merit(d);

      for (int i = 0; i < iterations; ++i)
      {
        if (value_or_inf(fc) < value_or_inf(fd))
        {
          b = d;
          d = c;
          fd = fc;
          c = b - inv_phi * (b - a);
          fc = merit(c);
        }
        else
        {
          a = c;
          c = d;
          fc = fd;
          d = a + inv_phi * (b - a);
          fd = merit(d);
        }
      }

      const double best_alpha = 0.5 * (a + b);
      return {best_alpha, merit(best_alpha)};
    }

    template <typename Merit>
    void write_csv(const Merit& merit, int trajectory_id, unsigned int local_newton_iter,
        double accepted_alpha)
    {
      constexpr double alpha_min = 0.0;
      constexpr double alpha_max = 20.0;
      // Most trajectories' interesting behaviour (the Newton step, the failure boundary, and the
      // optimum for the common case) sits within a unit or two of alpha=1; only some iterations'
      // true minimum lands much further out (observed up to ~alpha=12). Sampling that whole range
      // at one uniform density would either waste resolution on the near-empty tail or -- as it
      // did before this two-tier split -- blur out the near-alpha=1 region every panel actually
      // zooms into. So: dense near the start, coarser further out where only the broad shape (not
      // fine detail) matters.
      constexpr double dense_upper = 2.0;
      constexpr int dense_samples = 400;
      constexpr int coarse_samples = 200;

      const int sequence = sequence_counter_.fetch_add(1, std::memory_order_relaxed);
      const std::filesystem::path file_path =
          *output_dir_ / std::format("merit_curve_traj{}_iter{}_{:04d}.csv", trajectory_id,
                             local_newton_iter, sequence);

      std::ofstream out(file_path);
      out << "trajectory_id,local_newton_iter,alpha,merit,point_type,has_error,error_type\n";

      // point_type: "sample" (regular grid point), "accepted" (the step the line search actually
      // took -- for the plain, unsafeguarded loop this always coincides with "full_newton"),
      // "full_newton" (alpha=1 sampled exactly -- the grid above never lands exactly on 1, and
      // near a sharp minimum a nearest-neighbor lookup can be off by orders of magnitude),
      // "best" (the true minimizer via golden-section search -- same precision concern as above,
      // but for wherever the actual optimum lands rather than specifically at alpha=1).
      const auto write_row =
          [&](double alpha,
              const LocalNewtonLineSearch::MeritResult<ViscoplastUtils::ErrorType>& result,
              const char* point_type)
      {
        out << trajectory_id << ',' << local_newton_iter << ',' << alpha << ',';
        if (result.error.has_value())
          out << "," << point_type << ",1," << EnumTools::enum_name(*result.error) << '\n';
        else
          out << result.value << ',' << point_type << ",0,\n";
      };

      const double dense_hi = std::min(alpha_max, dense_upper);
      const double dense_spacing = (dense_hi - alpha_min) / (dense_samples - 1);
      const double coarse_spacing =
          alpha_max > dense_upper ? (alpha_max - dense_upper) / coarse_samples : dense_spacing;

      double best_sample_alpha = alpha_min;
      double best_sample_value = std::numeric_limits<double>::infinity();
      const auto sample_and_track = [&](double alpha)
      {
        const auto result = merit(alpha);
        write_row(alpha, result, "sample");
        if (!result.error.has_value() && result.value < best_sample_value)
        {
          best_sample_value = result.value;
          best_sample_alpha = alpha;
        }
      };

      for (int i = 0; i < dense_samples; ++i)
        sample_and_track(alpha_min + (dense_hi - alpha_min) * i / (dense_samples - 1));
      // i starts at 1: dense_hi was already sampled as the last dense point above.
      for (int i = 1; i <= coarse_samples && alpha_max > dense_upper; ++i)
        sample_and_track(dense_upper + (alpha_max - dense_upper) * i / coarse_samples);

      write_row(1.0, merit(1.0), "full_newton");
      if (std::isfinite(accepted_alpha))
        write_row(accepted_alpha, merit(accepted_alpha), "accepted");

      // Refine the coarse grid's best sample to full precision, searching only a tight window
      // around it (see golden_section_minimize's comment for why the full [alpha_min, alpha_max]
      // range isn't safe to search directly). Which tier's spacing applies depends on where the
      // best sample landed.
      const double grid_spacing = best_sample_alpha <= dense_hi ? dense_spacing : coarse_spacing;
      const auto [best_alpha, best_result] =
          golden_section_minimize(merit, std::max(alpha_min, best_sample_alpha - grid_spacing),
              std::min(alpha_max, best_sample_alpha + grid_spacing));
      write_row(best_alpha, best_result, "best");

      // The minimum found above is often a razor-thin cusp the coarse grid above steps right
      // over (its nearest point can be orders of magnitude short of the true minimum), which
      // would otherwise leave the plotted curve visibly disconnected from the "best" point.
      // Densely resample a narrow window around it (still marked "sample") so a plotted line
      // actually traces down into the cusp instead of jumping straight to a floating marker.
      constexpr int fine_samples = 60;
      const double fine_lo = std::max(alpha_min, best_alpha - 2.0 * grid_spacing);
      const double fine_hi = std::min(alpha_max, best_alpha + 2.0 * grid_spacing);
      for (int i = 0; i <= fine_samples; ++i)
      {
        const double alpha = fine_lo + (fine_hi - fine_lo) * i / fine_samples;
        write_row(alpha, merit(alpha), "sample");
      }
    }

    std::optional<std::filesystem::path> output_dir_;
    std::array<std::atomic<int>, kMaxTrackedIter> budgets_;
    std::atomic<int> sequence_counter_{0};
    std::atomic<int> trajectory_counter_{0};
  };
}  // namespace MeritExportDebug

FOUR_C_NAMESPACE_CLOSE

#endif
