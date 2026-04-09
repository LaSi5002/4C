// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_RUNTIME_VISUALIZATION_OUTPUT_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_RUNTIME_VISUALIZATION_OUTPUT_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_io_visualization_parameters.hpp"

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  class BeamContactRuntimeVisualizationOutputParams
  {
   public:
    //! constructor
    explicit BeamContactRuntimeVisualizationOutputParams(double restart_time);

    //! destructor
    virtual ~BeamContactRuntimeVisualizationOutputParams() = default;

    /**
     * \brief Return the container holding the general output parameters
     */
    const Core::IO::VisualizationParameters& get_visualization_parameters() const
    {
      return visualization_parameters_;
    }

    /// output interval regarding steps: write output every INTERVAL_STEPS steps
    int output_interval_in_steps() const { return output_interval_steps_; };

    /// whether to write output in every iteration of the nonlinear solver
    bool output_every_iteration() const { return output_every_iteration_; };

    /// whether to write output for contact forces
    bool is_write_contact_forces() const { return output_forces_; };

    /// whether to write output for gaps
    bool is_write_gaps() const { return output_gaps_; };

    /// whether to write output for contact angles
    bool is_write_angles() const { return output_angles_; };

    /// whether to write which contact contribution (or formulation) is active
    bool is_write_types() const { return output_types_; };


   private:
    //! General visualization parameters
    Core::IO::VisualizationParameters visualization_parameters_;

    /// output interval regarding steps: write output every INTERVAL_STEPS steps
    int output_interval_steps_;

    /// whether to write output in every iteration of the nonlinear solver
    bool output_every_iteration_;

    /// whether to write forces
    bool output_forces_;

    /// whether to write gaps
    bool output_gaps_;

    /// whether to write contact angles
    bool output_angles_;

    /// whether to write which contact contribution (or formulation) is active
    bool output_types_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
