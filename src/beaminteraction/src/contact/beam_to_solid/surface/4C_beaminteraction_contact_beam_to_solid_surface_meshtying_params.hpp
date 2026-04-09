// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_SURFACE_MESHTYING_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_SURFACE_MESHTYING_PARAMS_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_contact_beam_to_solid_params_base.hpp"

FOUR_C_NAMESPACE_OPEN


// Forward declaration.
namespace BeamInteraction
{
  class BeamToSolidSurfaceVisualizationOutputParams;
}

namespace BeamInteraction
{
  /**
   * \brief Class for beam to solid meshtying parameters.
   */
  class BeamToSolidSurfaceMeshtyingParams : public BeamToSolidParamsBase
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamToSolidSurfaceMeshtyingParams();

    /**
     * \brief Returns the coupling type for beam-to-surface coupling.
     */
    inline BeamToSolid::BeamToSolidSurfaceCoupling get_coupling_type() const
    {
      return coupling_type_;
    }

    /**
     * \brief Returns true if the coupling should be evaluated with FAD.
     */
    inline bool get_is_fad() const override
    {
      switch (coupling_type_)
      {
        case BeamToSolid::BeamToSolidSurfaceCoupling::reference_configuration_forced_to_zero:
        case BeamToSolid::BeamToSolidSurfaceCoupling::displacement:
          return false;
          break;
        case BeamToSolid::BeamToSolidSurfaceCoupling::reference_configuration_forced_to_zero_fad:
        case BeamToSolid::BeamToSolidSurfaceCoupling::displacement_fad:
        case BeamToSolid::BeamToSolidSurfaceCoupling::consistent_fad:
          return true;
          break;
        default:
          FOUR_C_THROW("Wrong coupling type.");
          break;
      }
      return false;
    }

    /**
     * \brief Returns the order of the FAD type.
     */
    inline int get_fad_order() const override
    {
      if (get_is_fad())
        return 2;
      else
        return 0;
    }

    /**
     * \brief Returns if rotational coupling is activated.
     */
    inline bool get_is_rotational_coupling() const { return rotational_coupling_; }

    /**
     * \brief Returns the penalty parameter for rotational coupling.
     */
    inline double get_rotational_coupling_penalty_parameter() const
    {
      return rotational_coupling_penalty_parameter_;
    }

    /**
     * \brief Returns the type of surface triad construction for rotational coupling.
     */
    inline BeamToSolid::BeamToSolidSurfaceRotationCoupling get_surface_triad_construction() const
    {
      return rotational_coupling_triad_construction_;
    }

    /**
     * \brief Returns a pointer to the visualization output parameters.
     * @return Pointer to visualization output parameters.
     */
    std::shared_ptr<BeamToSolidSurfaceVisualizationOutputParams>
    get_visualization_output_params_ptr();

   private:
    //! How the coupling should be evaluated.
    BeamToSolid::BeamToSolidSurfaceCoupling coupling_type_;

    //! Pointer to the visualization output parameters for beam to solid volume meshtying.
    std::shared_ptr<BeamToSolidSurfaceVisualizationOutputParams> output_params_ptr_;

    //! Penalty parameter for rotational coupling.
    double rotational_coupling_penalty_parameter_;

    //! Type of surface triad construction.
    BeamToSolid::BeamToSolidSurfaceRotationCoupling rotational_coupling_triad_construction_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
