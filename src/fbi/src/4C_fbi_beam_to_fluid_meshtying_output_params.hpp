// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_OUTPUT_PARAMS_HPP
#define FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_OUTPUT_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_contact_beam_to_solid_volume_meshtying_visualization_output_params.hpp"

FOUR_C_NAMESPACE_OPEN


namespace FBI
{
  class BeamToFluidMeshtyingVtkOutputParams
      : public BeamInteraction::BeamToSolidVolumeMeshtyingVisualizationOutputParams
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamToFluidMeshtyingVtkOutputParams();

    // returns flag enabling/disabling output of the constraint violation
    bool get_constraint_violation_output_flag() const { return constraint_violation_; };

   private:
    // bool enabling/disabling output of the constraint_enforcer violation
    bool constraint_violation_;
  };
}  // namespace FBI

FOUR_C_NAMESPACE_CLOSE

#endif
