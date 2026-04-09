// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_PARAMS_BASE_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SOLID_PARAMS_BASE_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_contact_beam_to_solid_input.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  /**
   * \brief Base class for beam to solid parameters.
   */
  class BeamToSolidParamsBase
  {
   public:
    /**
     * \brief Constructor.
     */
    explicit BeamToSolidParamsBase(const Teuchos::ParameterList& beam_to_solid_params_list);

    /**
     * \brief Destructor.
     */
    virtual ~BeamToSolidParamsBase() = default;

    /**
     * \brief Returns the contact discretization method.
     */
    inline BeamToSolid::BeamToSolidConstraintEnforcement get_constraint_enforcement() const
    {
      return constraint_enforcement_;
    }

    /**
     * \brief Returns constraints enforcement strategy.
     */
    inline BeamToSolid::BeamToSolidContactDiscretization get_contact_discretization() const
    {
      return contact_discretization_;
    }

    /**
     * \brief Returns the type of lagrange formulation used.
     */
    inline BeamToSolid::BeamToSolidLagrangeFormulation get_lagrange_formulation() const
    {
      return lagrange_formulation_;
    }

    /**
     * \brief Returns the shape function for the mortar Lagrange-multiplicators.
     */
    inline BeamToSolid::BeamToSolidMortarShapefunctions get_mortar_shape_function_type() const
    {
      return mortar_shape_function_;
    }

    /**
     * \brief Returns the penalty parameter.
     * @return penalty parameter.
     */
    inline double get_penalty_parameter() const { return penalty_parameter_; }

    /**
     * \brief Returns the augmentation scaling parameter for the beam DOFs.
     */
    [[nodiscard]] inline double get_augmentation_scaling_parameter_beam() const
    {
      return augmentation_scaling_parameter_beam_;
    }

    /**
     * \brief Returns the augmentation scaling parameter for the solid DOFs.
     */
    [[nodiscard]] inline double get_augmentation_scaling_parameter_solid() const
    {
      return augmentation_scaling_parameter_solid_;
    }

    /**
     * \brief Returns the Gauss rule.
     * @return gauss rule.
     */
    inline Core::FE::GaussRule1D get_gauss_rule() const { return gauss_rule_; }

    /**
     * \brief Returns true if the coupling should be evaluated with FAD.
     */
    virtual inline bool get_is_fad() const { return false; }

    /**
     * \brief Returns the order for the FAD type.
     */
    virtual inline int get_fad_order() const { return 0; }

    /**
     * \brief Returns the order for the FAD type.
     */
    [[nodiscard]] bool is_rotational_coupling() const { return rotational_coupling_; }

   protected:
    //! Enforcement strategy for constraints.
    BeamToSolid::BeamToSolidConstraintEnforcement constraint_enforcement_;

    //! discretization used for the contact.
    BeamToSolid::BeamToSolidContactDiscretization contact_discretization_;

    //! Shape function for the mortar Lagrange-multiplicators
    BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function_;

    //! Penalty parameter.
    double penalty_parameter_;

    /*! \brief Augmentation scaling parameters for beam-solid coupling.
     *
     * These parameters scale the penalty contributions in the augmented Lagrangian formulation
     * used for beam-solid coupling.
     *
     * augmentation_scaling_parameter_beam_: Scaling factor applied to the beam-side contribution.
     * augmentation_scaling_parameter_solid_: Scaling factor applied to the solid-side contribution.
     *
     * Using separate scaling parameters allows balancing the influence of beam and
     * solid discretizations, which may differ significantly in stiffness and dimensionality.
     * Proper tuning improves numerical conditioning, convergence behavior, and robustness
     * of the coupling algorithm.
     */
    double augmentation_scaling_parameter_beam_;
    double augmentation_scaling_parameter_solid_;

    //! Gauss rule to be used.
    Core::FE::GaussRule1D gauss_rule_;

    //! Flag for rotational coupling
    bool rotational_coupling_;

    //! Type of lagrange formulation
    BeamToSolid::BeamToSolidLagrangeFormulation lagrange_formulation_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
