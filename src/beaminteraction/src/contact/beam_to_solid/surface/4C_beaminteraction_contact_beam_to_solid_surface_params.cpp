// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_contact_beam_to_solid_surface_params.hpp"

#include "4C_beaminteraction_contact_beam_to_solid_surface_visualization_output_params.hpp"
#include "4C_geometry_pair_input.hpp"
#include "4C_global_data.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToSolidSurfaceContactParams::BeamToSolidSurfaceContactParams()
    : BeamToSolidParamsBase(
          Global::Problem::instance()->beam_interaction_params().sublist(
              "BEAM TO SOLID SURFACE CONTACT")),
      contact_type_(BeamToSolid::BeamToSolidSurfaceContact::none),
      penalty_law_data_{},
      mortar_contact_configuration_(BeamToSolid::BeamToSolidSurfaceContactMortarDefinedIn::none),
      output_params_ptr_(nullptr)
{
  // Teuchos parameter list for beam contact
  const Teuchos::ParameterList& beam_to_solid_contact_params_list =
      Global::Problem::instance()->beam_interaction_params().sublist(
          "BEAM TO SOLID SURFACE CONTACT");

  // Get parameters form input file.
  {
    contact_type_ = Teuchos::getIntegralValue<BeamToSolid::BeamToSolidSurfaceContact>(
        beam_to_solid_contact_params_list, "CONTACT_TYPE");

    penalty_law_data_.type =
        Teuchos::getIntegralValue<BeamToSolid::BeamToSolidSurfaceContactPenaltyLaw>(
            beam_to_solid_contact_params_list, "PENALTY_LAW");
    penalty_law_data_.penalty_parameter =
        beam_to_solid_contact_params_list.get<double>("PENALTY_PARAMETER");
    penalty_law_data_.penalty_parameter_g0 =
        beam_to_solid_contact_params_list.get<double>("PENALTY_PARAMETER_G0");

    mortar_contact_configuration_ =
        Teuchos::getIntegralValue<BeamToSolid::BeamToSolidSurfaceContactMortarDefinedIn>(
            beam_to_solid_contact_params_list, "MORTAR_CONTACT_DEFINED_IN");
  }

  // Setup the output parameter object.
  {
    output_params_ptr_ = std::make_shared<BeamToSolidSurfaceVisualizationOutputParams>();
  }
}


/**
 *
 */
int BeamInteraction::BeamToSolidSurfaceContactParams::get_fad_order() const

{
  switch (get_contact_type())
  {
    case BeamToSolid::BeamToSolidSurfaceContact::gap_variation:
      return 1;
      break;
    case BeamToSolid::BeamToSolidSurfaceContact::potential:
      return 2;
      break;
    default:
      FOUR_C_THROW("Got unexpected contact type.");
      return 0;
  }
}

FOUR_C_NAMESPACE_CLOSE
