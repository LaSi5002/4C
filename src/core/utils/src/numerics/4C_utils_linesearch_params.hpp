// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_LINESEARCH_PARAMS_HPP
#define FOUR_C_UTILS_LINESEARCH_PARAMS_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils::LineSearch
{
  enum class LineSearchType
  {
    none,
    error_adaptive_newton,
    armijo_backtracking,
    grippo_lampariello_lucidi_backtracking,
    golden_section
  };

  struct StepControlParams
  {
    double alpha_init;
    int max_iter;
  };

  struct ArmijoParams
  {
    double c1;
  };

  struct BacktrackingParams
  {
    int max_line_search_iterations;
  };

  struct GoldenSectionParams
  {
    double min_alpha;

    double interval_tol;
  };

  struct GrippoLamparielloLucidiParams
  {
    double rho;

    int max_history;
  };

}  // namespace Core::Utils::LineSearch

FOUR_C_NAMESPACE_CLOSE

#endif
