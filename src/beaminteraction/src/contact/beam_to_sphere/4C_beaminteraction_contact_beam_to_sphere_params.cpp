// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_contact_beam_to_sphere_params.hpp"

#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
BeamInteraction::BeamToSphereContactParams::BeamToSphereContactParams()
    : penalty_parameter_(-1.0)
{
  // Teuchos parameter list for beam contact
  const Teuchos::ParameterList& beam_to_sphere_contact_params_list =
      Global::Problem::instance()->beam_interaction_params().sublist("BEAM TO SPHERE CONTACT");

  penalty_parameter_ = beam_to_sphere_contact_params_list.get<double>("PENALTY_PARAMETER");

  if (penalty_parameter_ < 0.0)
    FOUR_C_THROW("beam-to-sphere penalty parameter must not be negative!");
}

FOUR_C_NAMESPACE_CLOSE
