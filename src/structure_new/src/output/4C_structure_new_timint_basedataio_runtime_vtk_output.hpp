// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_RUNTIME_VTK_OUTPUT_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_RUNTIME_VTK_OUTPUT_HPP

#include "4C_config.hpp"

#include "4C_io_visualization_parameters.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    class StructureRuntimeOutputParams;
    class BeamRuntimeOutputParams;
  }  // namespace Elements
}  // namespace Discret


namespace Solid
{
  namespace TimeInt
  {
    /** \brief Input data container for output at runtime for the structural (time) integration
     *
     * */
    class ParamsRuntimeOutput
    {
     public:
      explicit ParamsRuntimeOutput(const Teuchos::ParameterList& IO_vtk_structure_paramslist);

      /// destructor
      virtual ~ParamsRuntimeOutput() = default;

      /// output interval regarding steps: write output every INTERVAL_STEPS steps
      int output_interval_in_steps() const { return output_interval_steps_; };

      [[nodiscard]] int output_step_offset() const { return output_step_offset_; }

      /// whether to write output in every iteration of the nonlinear solver
      bool output_every_iteration() const { return output_every_iteration_; };

      /// whether to write special output for structure elements
      bool output_structure() const { return output_structure_; };

      /// whether to write special output for structure elements
      bool output_beams() const { return output_beams_; };

      /// get the data container for parameters regarding beams
      std::shared_ptr<const Discret::Elements::StructureRuntimeOutputParams> get_structure_params()
          const
      {
        return params_runtime_output_structure_;
      };

      /// get the data container for parameters regarding beams
      std::shared_ptr<const Discret::Elements::BeamRuntimeOutputParams> get_beam_params() const
      {
        return params_runtime_output_beams_;
      };


     private:
      /// @name variables controlling output
      /// @{

      /// output interval regarding steps: write output every INTERVAL_STEPS steps
      int output_interval_steps_ = -1;

      /// An offset added to the current step to shift the steps to be written
      int output_step_offset_ = 0;

      /// whether to write output in every iteration of the nonlinear solver
      bool output_every_iteration_ = false;

      /// whether to write output for structural elements
      bool output_structure_ = false;

      /// whether to write special output for beam elements
      bool output_beams_ = false;

      /// data container for input parameters related to output of structure at runtime
      std::shared_ptr<Discret::Elements::StructureRuntimeOutputParams>
          params_runtime_output_structure_ = nullptr;

      /// data container for input parameters related to output of beams at runtime
      std::shared_ptr<Discret::Elements::BeamRuntimeOutputParams> params_runtime_output_beams_ =
          nullptr;

      //@}
    };

  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
