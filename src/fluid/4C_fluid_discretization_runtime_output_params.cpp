// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_discretization_runtime_output_params.hpp"

#include "4C_inpar_fluid.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Discret::Elements::FluidRuntimeOutputParams::FluidRuntimeOutputParams(
    const Teuchos::ParameterList& IO_fluid_paramslist)
{
  output_velocity_state_ = IO_fluid_paramslist.get<bool>("VELOCITY");
  output_pressure_state_ = IO_fluid_paramslist.get<bool>("PRESSURE");
  output_acceleration_state_ = IO_fluid_paramslist.get<bool>("ACCELERATION");
  output_displacement_state_ = IO_fluid_paramslist.get<bool>("DISPLACEMENT");
  output_gridvelocity_state_ = IO_fluid_paramslist.get<bool>("GRIDVELOCITY");
  output_element_owner_ = IO_fluid_paramslist.get<bool>("ELEMENT_OWNER");
  output_element_gid_ = IO_fluid_paramslist.get<bool>("ELEMENT_GID");
  output_node_gid_ = IO_fluid_paramslist.get<bool>("NODE_GID");
}

FOUR_C_NAMESPACE_CLOSE
