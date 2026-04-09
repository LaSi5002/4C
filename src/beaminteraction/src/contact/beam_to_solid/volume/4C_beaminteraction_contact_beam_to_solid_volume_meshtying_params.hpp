// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_VOLUME_MESHTYING_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_VOLUME_MESHTYING_PARAMS_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_contact_beam_to_solid_params_base.hpp"

FOUR_C_NAMESPACE_OPEN


// Forward declaration.
namespace BeamInteraction
{
  class BeamToSolidVolumeMeshtyingVisualizationOutputParams;
}


namespace BeamInteraction
{
  /**
   * \brief Class for beam to solid meshtying parameters.
   */
  class BeamToSolidVolumeMeshtyingParams : public BeamToSolidParamsBase
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamToSolidVolumeMeshtyingParams();

    /**
     * \brief Returns the number of integration points along the circumference of the beams cross
     * section.
     * @return Number of points.
     */
    inline unsigned int get_number_of_integration_points_circumference() const
    {
      return integration_points_circumference_;
    }

    /**
     * \brief Returns the number of Fourier modes for cross-section mortar coupling.
     */
    inline int get_number_of_fourier_modes() const { return n_fourier_modes_; }

    /**
     * \brief Returns the type of rotational coupling to be used.
     */
    inline BeamToSolid::BeamToSolidRotationCoupling get_rotational_coupling_type() const
    {
      return rotational_coupling_triad_construction_;
    }

    /**
     * \brief Returns the shape function for the mortar Lagrange-multiplicators to interpolate the
     * rotational coupling.
     */
    inline BeamToSolid::BeamToSolidMortarShapefunctions get_mortar_shape_function_rotation_type()
        const
    {
      return mortar_shape_function_rotation_;
    }

    /**
     * \brief Returns the penalty parameter for rotational coupling.
     */
    inline double get_rotational_coupling_penalty_parameter() const
    {
      return rotational_coupling_penalty_parameter_;
    }

    /**
     * \brief Returns a pointer to the visualization output parameters.
     * @return Pointer to visualization output parameters.
     */
    std::shared_ptr<BeamToSolidVolumeMeshtyingVisualizationOutputParams>
    get_visualization_output_params_ptr();

    /**
     * \brief Returns if the restart configuration should be coupled.
     */
    inline bool get_couple_restart_state() const { return couple_restart_state_; }

   private:
    //! Number of integration points along the circumferential direction in the cross section
    //! projection.
    unsigned int integration_points_circumference_;

    //! Number of Fourier modes for cross-section coupling
    int n_fourier_modes_;

    //! Type of rotational coupling.
    BeamToSolid::BeamToSolidRotationCoupling rotational_coupling_triad_construction_;

    //! Shape function for the mortar Lagrange-multiplicators
    BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function_rotation_;

    //! Penalty parameter for rotational coupling.
    double rotational_coupling_penalty_parameter_;

    //! Pointer to the visualization output parameters for beam to solid volume meshtying.
    std::shared_ptr<BeamToSolidVolumeMeshtyingVisualizationOutputParams> output_params_ptr_;

    //! If the coupling terms should be evaluated with the restart configuration.
    bool couple_restart_state_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
