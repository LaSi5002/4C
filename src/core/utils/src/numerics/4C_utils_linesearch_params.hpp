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
    armijo_backtracking,
    armijo_safeguarded_quadratic_backtracking,
    goldstein_simple_bisection,
    weak_wolfe_simple_bisection,
    weak_wolfe_more_thuente,
    strong_wolfe_simple_bisection,
    strong_wolfe_more_thuente,
    dai_kou_simple_bisection,
    grippo_lampariello_lucidi_backtracking,
    grippo_lampariello_lucidi_safeguarded_quadratic_backtracking,
    zhang_hager_nonmonotone_armijo_backtracking,
    zhang_hager_nonmonotone_armijo_safeguarded_quadratic_backtracking,
    hager_zhang
  };

  //! step control parameters for line search algorithms
  struct StepControlParams
  {
    double alpha_init;
    int max_iter;
  };

  //! Armijo condition settings.
  struct ArmijoParams
  {
    //! Armijo sufficient decrease constant
    double c1;
  };

  //! Goldstein condition settings.
  struct GoldsteinParams
  {
    //! Goldstein sufficient-decrease parameter, required to be in (0, 1/2)
    double rho;
  };

  //! (Weak or strong) Wolfe condition settings.
  struct WolfeParams
  {
    //! Wolfe sufficient decrease constant
    double c1;

    //! Wolfe curvature constant
    double c2;
  };

  //! Dai-Kou condition settings.
  struct DaiKouParams
  {
    //! function-value tolerance
    double epsilon;

    //! sufficient-decrease parameter
    double rho;

    //! curvature parameter
    double sigma;

    //! scale of the positive summable sequence relative to the initial directional derivative
    double eta_relative;
  };

  //! Grippo-Lampariello-Lucidi condition settings.
  struct GrippoLamparielloLucidiParams
  {
    //! sufficient-decrease parameter
    double rho;

    //! maximum number of previous merit values used by the nonmonotone condition
    int max_history;
  };

  //! Zhang-Hager nonmonotone Armijo condition settings.
  struct ZhangHagerNonmonotoneParams
  {
    //! sufficient-decrease parameter
    double rho;

    //! averaging parameter controlling the degree of nonmonotonicity
    double eta;
  };

  //! Hager-Zhang settings.
  struct HagerZhangParams
  {
    //! lower curvature parameter
    double delta;

    //! upper curvature parameter
    double sigma;

    //! function-value tolerance
    double epsilon;

    //! interpolation parameter
    double theta;

    //! interval reduction parameter
    double gamma;

    //! bracket expansion parameter
    double rho;
  };
}  // namespace Core::Utils::LineSearch

FOUR_C_NAMESPACE_CLOSE

#endif
