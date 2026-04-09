// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_DISCRETIZATION_RUNTIME_OUTPUT_PARAMS_HPP
#define FOUR_C_FLUID_DISCRETIZATION_RUNTIME_OUTPUT_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    /** \brief Input data container for output at runtime for Fluid*/
    class FluidRuntimeOutputParams
    {
     public:
      /// constructor
      FluidRuntimeOutputParams() = default;
      explicit FluidRuntimeOutputParams(const Teuchos::ParameterList& IO_fluid_paramslist);

      /// destructor
      virtual ~FluidRuntimeOutputParams() = default;

      /// whether to write velocity
      bool output_velocity_state() const { return output_velocity_state_; };

      /// whether to write pressure
      bool output_pressure_state() const { return output_pressure_state_; };

      /// whether to write velocity
      bool output_acceleration_state() const { return output_acceleration_state_; };

      /// whether to write displacements
      bool output_displacement_state() const { return output_displacement_state_; };

      /// whether to write velocity
      bool output_grid_velocity_state() const { return output_gridvelocity_state_; };

      /// whether to write the owner of elements
      bool output_element_owner() const { return output_element_owner_; };

      /// whether to write the GIDs of elements
      bool output_element_gid() const { return output_element_gid_; };

      /// whether to write the GIDs of the nodes
      bool output_node_gid() const { return output_node_gid_; };


     private:
      /// @name variables controlling output
      /// @{

      /// whether to write velocity output
      bool output_velocity_state_ = false;

      /// whether to write pressure output
      bool output_pressure_state_ = false;

      /// whether to write acceleration output
      bool output_acceleration_state_ = false;

      /// whether to write displacement output
      bool output_displacement_state_ = false;

      /// whether to write grid velocity output
      bool output_gridvelocity_state_ = false;

      /// whether to write the owner of elements
      bool output_element_owner_ = false;

      /// whether to write the element GIDs
      bool output_element_gid_ = false;

      /// whether to write the node GIDs
      bool output_node_gid_ = false;
      //@}
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
