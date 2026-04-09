// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_RUNTIME_VTP_OUTPUT_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_RUNTIME_VTP_OUTPUT_HPP


#include "4C_config.hpp"

#include "4C_inpar_IO_runtime_vtp_output_structure.hpp"
#include "4C_io_visualization_parameters.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace TimeInt
  {
    /** \brief Input data container for VTP output at runtime for the structural (time) integration
     *
     * */
    class ParamsRuntimeVtpOutput
    {
     public:
      explicit ParamsRuntimeVtpOutput(const Teuchos::ParameterList& IO_vtp_structure_paramslist);

      /// destructor
      virtual ~ParamsRuntimeVtpOutput() = default;

      /// whether to write owner at visualization point
      bool output_owner() const { return output_owner_; };

      /// whether to write orientation at visualization point
      bool output_orientation_and_length() const { return output_orientationandlength_; };

      /// whether to write number of bonds at visualization point
      bool output_number_of_bonds() const { return output_numberofbonds_; };

      /// whether to write number of bonds at visualization point
      bool output_linking_force() const { return output_linkingforce_; };
      /*    /// whether to write displacements
          bool output_displacement_state() const
          {
            check_init_setup();
            return output_displacement_state_;
          };*/

      /*    /// get the data container for parameters regarding beams
          std::shared_ptr<const Discret::Elements::BeamRuntimeOutputParams> get_beam_params() const
          {
            check_init_setup();
            return params_runtime_output_beams_;
          };*/

     private:
      /// @name variables controlling output
      /// @{

      /// output interval regarding steps: write output every INTERVAL_STEPS steps
      int output_interval_steps_ = -1;

      /// whether to write output in every iteration of the nonlinear solver
      bool output_every_iteration_ = false;

      /// whether to write owner at visualization point
      bool output_owner_ = false;

      /// whether to write orientation at visualization point
      bool output_orientationandlength_ = false;

      /// whether to write number of bonds at visualization point
      bool output_numberofbonds_ = false;

      /// whether to write force acting on linker
      bool output_linkingforce_ = false;

      /*    /// whether to write displacement output
          bool output_displacement_state_;*/

      /*    /// data container for input parameters related to output of beams at runtime
          std::shared_ptr<Discret::Elements::BeamRuntimeOutputParams>
         params_runtime_output_beams_;*/

      //@}
    };

  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
