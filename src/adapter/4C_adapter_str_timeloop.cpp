// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_timeloop.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_structure_new_timint_base.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::StructureTimeLoop::StructureTimeLoop(
    Global::Problem& problem, std::shared_ptr<Structure> structure)
    : StructureWrapper(structure), problem_(problem)
{
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Adapter::StructureTimeLoop::integrate()
{
  // error checking variables
  Inpar::Solid::ConvergenceStatus convergencestatus = Inpar::Solid::conv_success;
  const Teuchos::ParameterList& rebalance_params =
      Global::Problem::instance()->structural_dynamic_params().sublist("DYNAMIC REBALANCE");
  const struct
  {
    bool enabled;
    int window_steps;
    int cooldown_steps;
    double imbalance_threshold;
  } rebalance_trigger = {
      .enabled = rebalance_params.get<bool>("ENABLED"),
      .window_steps = std::max(1, rebalance_params.get<int>("WINDOW_STEPS")),
      .cooldown_steps = std::max(0, rebalance_params.get<int>("COOLDOWN_STEPS")),
      .imbalance_threshold = rebalance_params.get<double>("IMBALANCE_THRESHOLD"),
  };
  std::deque<double> imbalance_history;
  int last_rebalance_step = std::numeric_limits<int>::min() / 2;

  auto maybe_rebalance = [&]()
  {
    if (!rebalance_trigger.enabled) return;

    auto* timint = dynamic_cast<Solid::TimeInt::Base*>(structure_.get());
    if (timint == nullptr) return;

    const std::vector<double> rank_eval_times = structure_->discretization()->get_rank_eval_times();
    if (rank_eval_times.empty()) return;

    const auto max_it = std::ranges::max_element(rank_eval_times);
    const double mean_eval_time =
        std::accumulate(rank_eval_times.begin(), rank_eval_times.end(), 0.0) /
        static_cast<double>(rank_eval_times.size());
    if (*max_it <= 1.0e-12 or mean_eval_time <= 1.0e-12) return;

    const double imbalance = *max_it / mean_eval_time;
    imbalance_history.push_back(imbalance);
    while (static_cast<int>(imbalance_history.size()) > rebalance_trigger.window_steps)
      imbalance_history.pop_front();

    if (static_cast<int>(imbalance_history.size()) < rebalance_trigger.window_steps) return;

    const double averaged_imbalance =
        std::accumulate(imbalance_history.begin(), imbalance_history.end(), 0.0) /
        static_cast<double>(imbalance_history.size());
    if (averaged_imbalance <= rebalance_trigger.imbalance_threshold) return;

    const int current_step = timint->get_step_n();
    if (current_step - last_rebalance_step < rebalance_trigger.cooldown_steps) return;

    Core::IO::cout << "====== Dynamic structure redistribution triggered after step "
                   << current_step << " (rolling imbalance " << averaged_imbalance << ", threshold "
                   << rebalance_trigger.imbalance_threshold << ")" << Core::IO::endl;

    if (timint->perform_dynamic_rebalance())
    {
      last_rebalance_step = current_step;
      imbalance_history.clear();
    }
  };

  // target time #timen_ and step #stepn_ already set
  // time loop
  while (not_finished() and (convergencestatus == Inpar::Solid::conv_success or
                                convergencestatus == Inpar::Solid::conv_fail_repeat))
  {
    // call the predictor
    pre_predict();
    prepare_time_step();

    // integrate time step, i.e. do corrector steps
    // after this step we hold disn_, etc
    pre_solve();
    convergencestatus = solve();

    // if everything is fine
    if (convergencestatus == Inpar::Solid::conv_success)
    {
      // calculate stresses, strains and energies
      // note: this has to be done before the update since otherwise a potential
      // material history is overwritten
      constexpr bool force_prepare = false;
      prepare_output(force_prepare);

      // update displacements, velocities, accelerations
      // after this call we will have disn_==dis_, etc
      // update time and step
      // update everything on the element level
      pre_update();
      update();
      post_update();

      // write output
      output();
      post_output();

      // print info about finished time step
      print_step();

      maybe_rebalance();
    }
    // todo: remove this as soon as old structure time integration is gone
    else if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(
                 problem_.structural_dynamic_params(), "INT_STRATEGY") == Inpar::Solid::int_old)
    {
      convergencestatus =
          perform_error_action(convergencestatus);  // something went wrong update error code
                                                    // according to chosen divcont action
    }
  }

  post_time_loop();

  // that's it say what went wrong
  return convergencestatus;
}

FOUR_C_NAMESPACE_CLOSE
