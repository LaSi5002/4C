// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SPHERE_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_BEAM_TO_SPHERE_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  class BeamToSphereContactParams
  {
   public:
    //! constructor
    BeamToSphereContactParams();

    //! destructor
    virtual ~BeamToSphereContactParams() = default;

    inline double beam_to_sphere_penalty_param() const { return penalty_parameter_; }

   private:
    //! beam-to-sphere penalty parameter
    double penalty_parameter_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
