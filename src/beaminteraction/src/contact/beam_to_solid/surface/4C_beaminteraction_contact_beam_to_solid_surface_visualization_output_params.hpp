// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_SURFACE_VISUALIZATION_OUTPUT_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_SURFACE_VISUALIZATION_OUTPUT_PARAMS_HPP


#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  class BeamToSolidSurfaceVisualizationOutputParams
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamToSolidSurfaceVisualizationOutputParams();

    /**
     * \brief Destructor.
     */
    virtual ~BeamToSolidSurfaceVisualizationOutputParams() = default;

    /**
     * \brief Return the output every iteration flag.
     */
    bool get_output_every_iteration() const { return output_every_iteration_; }

    /**
     * \brief Output interval regarding steps: write output every INTERVAL_STEPS steps.
     */
    int output_interval_in_steps() const
    {
      return output_interval_steps_;
    };

    /**
     * \brief Return the output flag.
     */
    bool get_output_flag() const { return output_flag_; }

    /**
     * \brief Return the nodal forces flag.
     */
    bool get_nodal_force_output_flag() const { return nodal_forces_; }

    /**
     * \brief Return the averaged nodal normals flag.
     */
    bool get_averaged_normals_output_flag() const { return averaged_normals_; }

    /**
     * \brief Return the mortar lambda discrete flag.
     */
    bool get_mortar_lambda_discret_output_flag() const { return mortar_lambda_discret_; }

    /**
     * \brief Return the mortar lambda continuous flag.
     */
    bool get_mortar_lambda_continuous_output_flag() const { return mortar_lambda_continuous_; }

    /**
     * \brief Return the number of segments for continuous mortar output.
     */
    unsigned int get_mortar_lambda_continuous_segments() const
    {
      return mortar_lambda_continuous_segments_;
    }

    /**
     * \brief Return the nodal forces flag.
     */
    bool get_segmentation_output_flag() const { return segmentation_; }

    /**
     * \brief Return the integration output flag.
     */
    bool get_integration_points_output_flag() const { return integration_points_; }

    /**
     * \brief Return the write unique IDs output flag.
     */
    bool get_write_unique_ids_flag() const { return write_unique_ids_; }
   protected:
    //! Output interval regarding steps: write output every INTERVAL_STEPS steps.
    int output_interval_steps_;

    //! Whether to write output in every iteration of the nonlinear solver.
    bool output_every_iteration_;

    //! Flag whether or not to write output.
    bool output_flag_;

    //! Flag whether or not to output resulting nodal forces.
    bool nodal_forces_;

    //! Flag whether or not to output averaged nodal normals.
    bool averaged_normals_;

    //! Flag whether or not to output resulting discrete Lagrange multiplier values for mortar
    //! pairs.
    bool mortar_lambda_discret_;

    //! Flag whether or not to output resulting continuous Lagrange multiplier function for mortar
    //! pairs.
    bool mortar_lambda_continuous_;

    //! Number of segments to use for the continuous mortar output.
    unsigned int mortar_lambda_continuous_segments_;

    //! Flag whether or not to write segmentation data.
    bool segmentation_;

    //! Flag whether or not to write data at integration points.
    bool integration_points_;

    //! Flag whether or not to write unique cell and point data IDs.
    bool write_unique_ids_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
