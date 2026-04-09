// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_data.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Solid::ModelEvaluator::BeamData::BeamData()
    : beta_(-1.0), gamma_(-1.0), alphaf_(-1.0), alpham_(-1.0)
{
  // empty constructor
}

FOUR_C_NAMESPACE_CLOSE
