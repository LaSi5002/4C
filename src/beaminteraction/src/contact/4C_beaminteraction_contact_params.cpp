// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_contact_params.hpp"

#include "4C_beaminteraction_contact_beam_to_beam_params.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_edge_params.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_surface_meshtying_params.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_surface_params.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_contact_beam_to_sphere_params.hpp"
#include "4C_beaminteraction_contact_runtime_visualization_output_params.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamContactParams::BeamContactParams()
    : beam_to_beam_contact_params_(nullptr),
      beam_to_sphere_contact_params_(nullptr),
      beam_to_solid_volume_meshtying_params_(nullptr),
      beam_to_solid_surface_meshtying_params_(nullptr),
      beam_to_solid_surface_contact_params_(nullptr),
      beam_contact_runtime_output_params_(nullptr)
{
  // empty constructor
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactParams::build_beam_to_beam_contact_params()
{
  beam_to_beam_contact_params_ = std::make_shared<BeamInteraction::BeamToBeamContactParams>();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactParams::build_beam_contact_runtime_output_params(
    const double restart_time)
{
  beam_contact_runtime_output_params_ =
      std::make_shared<BeamInteraction::BeamContactRuntimeVisualizationOutputParams>(restart_time);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactParams::build_beam_to_solid_edge_contact_params()
{
  beam_to_solid_edge_contact_params_ = initialize_validate_beam_to_edge_contact_params();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactParams::build_beam_to_sphere_contact_params()
{
  beam_to_sphere_contact_params_ = std::make_shared<BeamInteraction::BeamToSphereContactParams>();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactParams::build_beam_to_solid_volume_meshtying_params()
{
  beam_to_solid_volume_meshtying_params_ =
      std::make_shared<BeamInteraction::BeamToSolidVolumeMeshtyingParams>();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactParams::build_beam_to_solid_surface_meshtying_params()
{
  beam_to_solid_surface_meshtying_params_ =
      std::make_shared<BeamInteraction::BeamToSolidSurfaceMeshtyingParams>();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactParams::build_beam_to_solid_surface_contact_params()
{
  beam_to_solid_surface_contact_params_ =
      std::make_shared<BeamInteraction::BeamToSolidSurfaceContactParams>();
}

FOUR_C_NAMESPACE_CLOSE
