// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_DISCRETIZATION_RUNTIME_OUTPUT_PARAMS_HPP
#define FOUR_C_BEAM3_DISCRETIZATION_RUNTIME_OUTPUT_PARAMS_HPP


#include "4C_config.hpp"

#include "4C_beam3_discretization_runtime_output_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    /** \brief Input data container for output at runtime for beams
     *
     * */
    class BeamRuntimeOutputParams
    {
     public:
      /// constructor
      BeamRuntimeOutputParams() = default;
      explicit BeamRuntimeOutputParams(
          const Teuchos::ParameterList& IO_vtk_structure_beams_paramslist);

      /// destructor
      virtual ~BeamRuntimeOutputParams() = default;


      /// whether to write displacements
      bool output_displacement_state() const { return output_displacement_state_; };

      /// whether to write triads at the Gauss points
      bool is_write_triad_visualization_points() const
      {
        return write_triads_visualizationpoints_;
      };

      /// whether to use absolute positions or initial positions for the vtu geometry definition
      /// (i.e. for the visualization point coordinates)
      bool use_absolute_positions() const
      {
        return use_absolute_positions_visualizationpoint_coordinates_;
      };

      /// whether to write material cross-section strains at the Gauss points
      bool is_write_internal_energy_element() const { return write_internal_energy_element_; };

      /// whether to write material cross-section strains at the Gauss points
      bool is_write_kinetic_energy_element() const { return write_kinetic_energy_element_; };

      /// whether to write material cross-section strains at the Gauss points
      bool is_write_material_strains_gauss_points() const
      {
        return write_material_crosssection_strains_gausspoints_;
      };

      /// whether to write material cross-section strains at the visualization points
      bool is_write_material_strains_continuous() const
      {
        return write_material_crosssection_strains_continuous_;
      };

      /// whether to write material cross-section stresses at the Gauss points
      bool is_write_material_stresses_gauss_points() const
      {
        return write_material_crosssection_stresses_gausspoints_;
      };

      /// whether to write material cross-section stresses at the visualization points
      bool is_write_material_stress_continuous() const
      {
        return write_material_crosssection_strains_continuous_;
      };

      /// whether to write material cross-section stresses at the Gauss points
      bool is_write_spatial_stresses_gauss_points() const
      {
        return write_spatial_crosssection_stresses_gausspoints_;
      };

      /// whether to write material cross-section stresses at the Gauss points
      bool is_write_element_filament_condition() const { return write_filament_condition_; };

      /// whether to write element and network orientation parameter
      bool is_write_orientation_parameter() const { return write_orientation_parameter_; };

      /// whether to write crosssection forces of periodic rve in x, y, and z direction
      bool is_write_rve_crosssection_forces() const { return write_rve_crosssection_forces_; };

      /// whether to write the element reference length
      bool is_write_ref_length() const { return write_ref_length_; };

      /// whether to write beam element GIDs.
      bool is_write_element_gid() const { return write_element_gid_; };

      /// write ghosting information
      bool is_write_element_ghosting() const { return write_element_ghosting_; };

      /// number of visualization subsegments.
      unsigned int get_number_visualization_subsegments() const { return n_subsegments_; };


     private:
      /// @name variables controlling output
      /// @{

      /// whether to write displacement output
      bool output_displacement_state_ = false;

      /// whether to use absolute positions or initial positions for the vtu geometry definition
      /// (i.e. for the visualization point coordinates)
      bool use_absolute_positions_visualizationpoint_coordinates_ = true;

      /// whether to write internal (elastic) energy for each element
      bool write_internal_energy_element_ = false;

      /// whether to write kinetic energy for each element
      bool write_kinetic_energy_element_ = false;

      /// whether to write triads at the visualization points
      bool write_triads_visualizationpoints_ = false;

      /// whether to write material cross-section strains at the Gauss points
      bool write_material_crosssection_strains_gausspoints_ = false;

      /// whether to write material cross-section strains at the visualization points
      bool write_material_crosssection_strains_continuous_ = false;

      /// whether to write material cross-section stresses at the Gauss points
      bool write_material_crosssection_stresses_gausspoints_ = false;

      /// whether to write spatial cross-section stresses at the Gauss points
      bool write_spatial_crosssection_stresses_gausspoints_ = false;

      /// whether to write beam filament condition (id, type)
      bool write_filament_condition_ = false;

      /// whether to write element and network orientation parameter
      bool write_orientation_parameter_ = false;

      /// whether to write crosssection forces of periodic rve in x, y, and z direction
      bool write_rve_crosssection_forces_ = false;

      /// whether to write the element GIDs.
      bool write_ref_length_ = false;

      /// whether to write the element GIDs.
      bool write_element_gid_ = false;

      /// whether to write the element ghosting information.
      bool write_element_ghosting_ = false;

      /// number of visualization subsegments
      unsigned int n_subsegments_ = 0;

      //@}
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
