// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_discretization_runtime_output_params.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Discret::Elements::StructureRuntimeOutputParams::StructureRuntimeOutputParams(
    const Teuchos::ParameterList& IO_vtk_structure_structure_paramslist)
{
  output_displacement_state_ = IO_vtk_structure_structure_paramslist.get<bool>("DISPLACEMENT");
  output_velocity_state_ = IO_vtk_structure_structure_paramslist.get<bool>("VELOCITY");
  output_acceleration_state_ = IO_vtk_structure_structure_paramslist.get<bool>("ACCELERATION");
  output_contact_ = IO_vtk_structure_structure_paramslist.get<bool>("OUTPUT_CONTACT");
  output_element_owner_ = IO_vtk_structure_structure_paramslist.get<bool>("ELEMENT_OWNER");
  output_element_gid_ = IO_vtk_structure_structure_paramslist.get<bool>("ELEMENT_GID");
  output_element_material_id_ = IO_vtk_structure_structure_paramslist.get<bool>("ELEMENT_MAT_ID");
  output_element_ghosting_ = IO_vtk_structure_structure_paramslist.get<bool>("ELEMENT_GHOSTING");
  output_optional_quantity_ =
      IO_vtk_structure_structure_paramslist.get<Inpar::Solid::OptQuantityType>("OPTIONAL_QUANTITY");
  output_node_gid_ = IO_vtk_structure_structure_paramslist.get<bool>("NODE_GID");
  output_stress_strain_ = IO_vtk_structure_structure_paramslist.get<bool>("STRESS_STRAIN");
  gauss_point_data_output_type_ = Teuchos::getIntegralValue<Inpar::Solid::GaussPointDataOutputType>(
      IO_vtk_structure_structure_paramslist, "GAUSS_POINT_DATA_OUTPUT_TYPE");

  if (output_stress_strain_)
  {
    // If stress / strain data should be output, check that the relevant parameters in the --IO
    // section are set.
    const Teuchos::ParameterList& io_parameter_list = Global::Problem::instance()->io_params();
    auto io_stress =
        Teuchos::getIntegralValue<Inpar::Solid::StressType>(io_parameter_list, "STRUCT_STRESS");
    auto io_strain =
        Teuchos::getIntegralValue<Inpar::Solid::StrainType>(io_parameter_list, "STRUCT_STRAIN");
    if (io_stress == Inpar::Solid::stress_none and io_strain == Inpar::Solid::strain_none)
    {
      FOUR_C_THROW(
          "If stress / strain runtime output is required, one or two of the flags STRUCT_STRAIN / "
          "STRUCT_STRESS in the --IO section has to be activated.");
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
