// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_contact_beam_to_solid_params_base.hpp"

#include "4C_geometry_pair_input.hpp"
#include "4C_global_data.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToSolidParamsBase::BeamToSolidParamsBase(
    const Teuchos::ParameterList& beam_to_solid_params_list)
    : constraint_enforcement_(BeamToSolid::BeamToSolidConstraintEnforcement::none),
      contact_discretization_(BeamToSolid::BeamToSolidContactDiscretization::none),
      mortar_shape_function_(BeamToSolid::BeamToSolidMortarShapefunctions::none),
      penalty_parameter_(-1.0),
      augmentation_scaling_parameter_beam_(-1.0),
      augmentation_scaling_parameter_solid_(-1.0),
      gauss_rule_(Core::FE::GaussRule1D::undefined),
      rotational_coupling_(false),
      lagrange_formulation_(BeamToSolid::BeamToSolidLagrangeFormulation::none)
{
  // Get parameters form input file.
  {
    // Constraint enforcement.
    constraint_enforcement_ =
        Teuchos::getIntegralValue<BeamToSolid::BeamToSolidConstraintEnforcement>(
            beam_to_solid_params_list, "CONSTRAINT_STRATEGY");

    //  Type of lagrange formulation used
    if (beam_to_solid_params_list.isParameter("LAGRANGE_FORMULATION"))
    {
      lagrange_formulation_ =
          Teuchos::getIntegralValue<BeamToSolid::BeamToSolidLagrangeFormulation>(
              beam_to_solid_params_list, "LAGRANGE_FORMULATION");
    }

    // Contact discretization to be used.
    contact_discretization_ =
        Teuchos::getIntegralValue<BeamToSolid::BeamToSolidContactDiscretization>(
            beam_to_solid_params_list, "CONTACT_DISCRETIZATION");

    // Contact discretization to be used.
    mortar_shape_function_ =
        Teuchos::getIntegralValue<BeamToSolid::BeamToSolidMortarShapefunctions>(
            beam_to_solid_params_list, "MORTAR_SHAPE_FUNCTION");

    // Penalty parameter.
    penalty_parameter_ = beam_to_solid_params_list.get<double>("PENALTY_PARAMETER");
    if (penalty_parameter_ < 0.0)
      FOUR_C_THROW("beam-to-volume-meshtying penalty parameter must not be negative!");

    if (beam_to_solid_params_list.isParameter("AUGMENTATION_SCALING_PARAMETER_BEAM") and
        beam_to_solid_params_list.isParameter("AUGMENTATION_SCALING_PARAMETER_SOLID"))
    {
      // Augmentation scaling parameters.
      augmentation_scaling_parameter_beam_ =
          beam_to_solid_params_list.get<double>("AUGMENTATION_SCALING_PARAMETER_BEAM");
      augmentation_scaling_parameter_solid_ =
          beam_to_solid_params_list.get<double>("AUGMENTATION_SCALING_PARAMETER_SOLID");
    }

    // Gauss rule for integration along the beam (segments).
    gauss_rule_ =
        GeometryPair::int_to_gauss_rule1_d(beam_to_solid_params_list.get<int>("GAUSS_POINTS"));
  }
}

FOUR_C_NAMESPACE_CLOSE
