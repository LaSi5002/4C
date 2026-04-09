// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_beam_to_fluid_meshtying_output_params.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_IO_runtime_output.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

FBI::BeamToFluidMeshtyingVtkOutputParams::BeamToFluidMeshtyingVtkOutputParams()
    : BeamInteraction::BeamToSolidVolumeMeshtyingVisualizationOutputParams(),
      constraint_violation_(false)
{
  // Teuchos parameter lists from input file.
  const Teuchos::ParameterList& beam_to_fluid_meshtying_visualization_output_paramslist =
      Global::Problem::instance()
          ->fbi_params()
          .sublist("BEAM TO FLUID MESHTYING")
          .sublist("RUNTIME VTK OUTPUT");
  const Teuchos::ParameterList& global_visualization_output_paramslist =
      Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT");

  // Get global parameters.
  output_interval_steps_ = global_visualization_output_paramslist.get<int>("INTERVAL_STEPS");
  output_every_iteration_ = global_visualization_output_paramslist.get<bool>("EVERY_ITERATION");

  // Get beam to fluid mesh tying specific parameters.
  output_flag_ = beam_to_fluid_meshtying_visualization_output_paramslist.get<bool>("WRITE_OUTPUT");

  nodal_forces_ = beam_to_fluid_meshtying_visualization_output_paramslist.get<bool>("NODAL_FORCES");

  segmentation_ = beam_to_fluid_meshtying_visualization_output_paramslist.get<bool>("SEGMENTATION");

  integration_points_ =
      beam_to_fluid_meshtying_visualization_output_paramslist.get<bool>("INTEGRATION_POINTS");

  constraint_violation_ =
      beam_to_fluid_meshtying_visualization_output_paramslist.get<bool>("CONSTRAINT_VIOLATION");
}

FOUR_C_NAMESPACE_CLOSE
