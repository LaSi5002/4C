// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_discretization_runtime_output_params.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Discret::Elements::BeamRuntimeOutputParams::BeamRuntimeOutputParams(
    const Teuchos::ParameterList& IO_vtk_structure_beams_paramslist)
{
  output_displacement_state_ = IO_vtk_structure_beams_paramslist.get<bool>("DISPLACEMENT");

  use_absolute_positions_visualizationpoint_coordinates_ =
      IO_vtk_structure_beams_paramslist.get<bool>("USE_ABSOLUTE_POSITIONS");

  write_internal_energy_element_ =
      IO_vtk_structure_beams_paramslist.get<bool>("INTERNAL_ENERGY_ELEMENT");

  write_kinetic_energy_element_ =
      IO_vtk_structure_beams_paramslist.get<bool>("KINETIC_ENERGY_ELEMENT");

  write_triads_visualizationpoints_ =
      IO_vtk_structure_beams_paramslist.get<bool>("TRIAD_VISUALIZATIONPOINT");

  write_material_crosssection_strains_gausspoints_ =
      IO_vtk_structure_beams_paramslist.get<bool>("STRAINS_GAUSSPOINT");

  write_material_crosssection_strains_continuous_ =
      IO_vtk_structure_beams_paramslist.get<bool>("STRAINS_CONTINUOUS");

  write_material_crosssection_stresses_gausspoints_ =
      IO_vtk_structure_beams_paramslist.get<bool>("MATERIAL_FORCES_GAUSSPOINT");

  write_material_crosssection_strains_continuous_ =
      IO_vtk_structure_beams_paramslist.get<bool>("MATERIAL_FORCES_CONTINUOUS");

  write_spatial_crosssection_stresses_gausspoints_ =
      IO_vtk_structure_beams_paramslist.get<bool>("SPATIAL_FORCES_GAUSSPOINT");

  write_orientation_parameter_ =
      IO_vtk_structure_beams_paramslist.get<bool>("ORIENTATION_PARAMETER");

  write_rve_crosssection_forces_ =
      IO_vtk_structure_beams_paramslist.get<bool>("RVE_CROSSSECTION_FORCES");

  write_ref_length_ = IO_vtk_structure_beams_paramslist.get<bool>("REF_LENGTH");

  write_element_gid_ = IO_vtk_structure_beams_paramslist.get<bool>("ELEMENT_GID");

  write_element_ghosting_ = IO_vtk_structure_beams_paramslist.get<bool>("ELEMENT_GHOSTING");

  n_subsegments_ = IO_vtk_structure_beams_paramslist.get<int>("NUMBER_SUBSEGMENTS");
  if (n_subsegments_ < 1)
    FOUR_C_THROW("The number of subsegments has to be at least 1. Got {}", n_subsegments_);
}

FOUR_C_NAMESPACE_CLOSE
