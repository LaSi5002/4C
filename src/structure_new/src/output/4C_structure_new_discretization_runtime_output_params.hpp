// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_DISCRETIZATION_RUNTIME_OUTPUT_PARAMS_HPP
#define FOUR_C_STRUCTURE_NEW_DISCRETIZATION_RUNTIME_OUTPUT_PARAMS_HPP


#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    /** \brief Input data container for output at runtime for structure*/
    class StructureRuntimeOutputParams
    {
     public:
      /// constructor
      StructureRuntimeOutputParams() = default;
      explicit StructureRuntimeOutputParams(
          const Teuchos::ParameterList& IO_vtk_structure_structure_paramslist);

      /// destructor
      virtual ~StructureRuntimeOutputParams() = default;

      /// whether to write displacements
      [[nodiscard]] bool output_displacement_state() const { return output_displacement_state_; }

      /// whether to write velocity
      [[nodiscard]] bool output_velocity_state() const { return output_velocity_state_; }

      /// whether to write acceleration
      [[nodiscard]] bool output_acceleration_state() const { return output_acceleration_state_; }

      /// whether to contact data
      [[nodiscard]] bool output_contact() const { return output_contact_; }


      /// whether to write the owner of elements
      [[nodiscard]] bool output_element_owner() const { return output_element_owner_; }

      /// whether to write the GIDs of elements
      [[nodiscard]] bool output_element_gid() const { return output_element_gid_; }

      /// whether to write the material IDs of elements
      [[nodiscard]] bool output_element_material_id() const { return output_element_material_id_; }

      /// whether to write the ghosting information of elements
      [[nodiscard]] bool output_element_ghosting() const { return output_element_ghosting_; }

      /// which optional solid output to write default: none
      [[nodiscard]] Inpar::Solid::OptQuantityType output_optional_quantity() const
      {
        return output_optional_quantity_;
      }

      /// whether to write the GIDs of the nodes
      [[nodiscard]] bool output_node_gid() const { return output_node_gid_; }

      /// whether to write stress and / or strain data
      [[nodiscard]] bool output_stress_strain() const { return output_stress_strain_; }

      /// Return output type of Gauss point data
      [[nodiscard]] Inpar::Solid::GaussPointDataOutputType gauss_point_data_output() const
      {
        return gauss_point_data_output_type_;
      }

     private:
      /// @name variables controlling output
      /// @{

      /// whether to write displacement output
      bool output_displacement_state_ = false;

      /// whether to write velocity output
      bool output_velocity_state_ = false;

      /// whether to write acceleration output
      bool output_acceleration_state_ = false;

      /// whether to write contact data
      bool output_contact_ = false;

      /// whether to write the owner of elements
      bool output_element_owner_ = false;

      /// whether to write the element GIDs
      bool output_element_gid_ = false;

      /// whether to write the element material IDs
      bool output_element_material_id_ = false;

      /// whether to write the element ghosting information
      bool output_element_ghosting_ = false;

      /// whether to write an optional quantity
      Inpar::Solid::OptQuantityType output_optional_quantity_ = Inpar::Solid::optquantity_none;

      /// whether to write the node GIDs
      bool output_node_gid_ = false;

      /// whether to write stress and / or strain data
      bool output_stress_strain_ = false;

      /// Output type of Gauss point data
      Inpar::Solid::GaussPointDataOutputType gauss_point_data_output_type_ =
          Inpar::Solid::GaussPointDataOutputType::none;
      //@}
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
