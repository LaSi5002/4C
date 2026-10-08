// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_mat_inelastic_defgrad_factors.hpp"

namespace
{
  using namespace FourC;
  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;
  namespace LineSearch = Core::Utils::LineSearch;

  ViscoplastUtils::LocalNewtonLineSearchRecoveryParams make_recovery_params(
      const ViscoplastUtils::RecoveryStrategy strategy,
      const ViscoplastUtils::LocalNewtonLineSearchIndividualContractionFactorParams& factors = {})
  {
    return ViscoplastUtils::LocalNewtonLineSearchRecoveryParams{
        .strategy = strategy, .individual_contraction_factor = factors};
  }

  TEST(VplastRecoveryPolicyTest, ViscoplasticErrorRecoveryAppliesAbortToEveryReachableError)
  {
    const auto recovery = ViscoplastUtils::viscoplastic_error_recovery(
        make_recovery_params(ViscoplastUtils::RecoveryStrategy::abort));

    for (const auto error : {ViscoplastUtils::ErrorType::overflow_error,
             ViscoplastUtils::ErrorType::negative_plastic_strain,
             ViscoplastUtils::ErrorType::failed_matrix_log_evaluation,
             ViscoplastUtils::ErrorType::failed_matrix_exp_evaluation})
    {
      EXPECT_TRUE(std::holds_alternative<LineSearch::AbortLineSearch>(recovery(error)))
          << "error " << EnumTools::enum_name(error);
    }
  }

  TEST(
      VplastRecoveryPolicyTest, ViscoplasticErrorRecoveryAppliesTreatAsTooHighToEveryReachableError)
  {
    const auto recovery = ViscoplastUtils::viscoplastic_error_recovery(
        make_recovery_params(ViscoplastUtils::RecoveryStrategy::treat_as_too_high));

    for (const auto error : {ViscoplastUtils::ErrorType::overflow_error,
             ViscoplastUtils::ErrorType::negative_plastic_strain,
             ViscoplastUtils::ErrorType::failed_matrix_log_evaluation,
             ViscoplastUtils::ErrorType::failed_matrix_exp_evaluation})
    {
      EXPECT_TRUE(std::holds_alternative<LineSearch::TreatAsTooHigh>(recovery(error)))
          << "error " << EnumTools::enum_name(error);
    }
  }

  TEST(VplastRecoveryPolicyTest, ViscoplasticErrorRecoveryUsesDistinctPerErrorFactors)
  {
    const ViscoplastUtils::LocalNewtonLineSearchIndividualContractionFactorParams factors{
        .overflow_error = 0.1,
        .negative_plastic_strain = 0.2,
        .failed_matrix_log_evaluation = 0.3,
        .failed_matrix_exp_evaluation = 0.4,
    };
    const auto recovery = ViscoplastUtils::viscoplastic_error_recovery(make_recovery_params(
        ViscoplastUtils::RecoveryStrategy::individual_contraction_factor, factors));

    const auto factor_for = [&recovery](const ViscoplastUtils::ErrorType error)
    {
      const LineSearch::RecoveryAction result = recovery(error);
      EXPECT_TRUE(std::holds_alternative<LineSearch::ContractStep>(result))
          << "error " << EnumTools::enum_name(error);
      return std::get<LineSearch::ContractStep>(result).factor;
    };

    EXPECT_DOUBLE_EQ(factor_for(ViscoplastUtils::ErrorType::overflow_error), 0.1);
    EXPECT_DOUBLE_EQ(factor_for(ViscoplastUtils::ErrorType::negative_plastic_strain), 0.2);
    EXPECT_DOUBLE_EQ(factor_for(ViscoplastUtils::ErrorType::failed_matrix_log_evaluation), 0.3);
    EXPECT_DOUBLE_EQ(factor_for(ViscoplastUtils::ErrorType::failed_matrix_exp_evaluation), 0.4);
  }

  TEST(VplastRecoveryPolicyTest,
      ViscoplasticErrorRecoveryWithIndividualContractionFactorFallsBackToAbortForUnlistedErrors)
  {
    const auto recovery = ViscoplastUtils::viscoplastic_error_recovery(
        make_recovery_params(ViscoplastUtils::RecoveryStrategy::individual_contraction_factor,
            {.overflow_error = 0.1,
                .negative_plastic_strain = 0.2,
                .failed_matrix_log_evaluation = 0.3,
                .failed_matrix_exp_evaluation = 0.4}));

    EXPECT_TRUE(std::holds_alternative<LineSearch::AbortLineSearch>(
        recovery(ViscoplastUtils::ErrorType::no_flow_resistance)));
  }
}  // namespace
