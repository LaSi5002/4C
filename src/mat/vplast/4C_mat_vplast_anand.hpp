// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VPLAST_ANAND_HPP
#define FOUR_C_MAT_VPLAST_ANAND_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Viscoplastic
  {
    namespace PAR
    {
      class Anand : public Core::Mat::PAR::Parameter
      {
       public:
        explicit Anand(const Core::Mat::PAR::Parameter::Data& matdata);

        std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; }

        [[nodiscard]] double strain_rate_pre_fac() const { return strain_rate_pre_fac_; }
        [[nodiscard]] double strain_rate_sensitivity() const { return strain_rate_sensitivity_; }
        [[nodiscard]] double init_flow_res() const { return init_flow_res_; }
        [[nodiscard]] double harden_rate_sensitivity() const { return harden_rate_sensitivity_; }
        [[nodiscard]] double harden_rate_pre_fac() const { return harden_rate_pre_fac_; }
        [[nodiscard]] double flow_res_sat_fac() const { return flow_res_sat_fac_; }
        [[nodiscard]] double flow_res_sat_exp() const { return flow_res_sat_exp_; }

       private:
        const double strain_rate_pre_fac_;
        const double strain_rate_sensitivity_;
        const double init_flow_res_;
        const double harden_rate_sensitivity_;
        const double harden_rate_pre_fac_;
        const double flow_res_sat_fac_;
        const double flow_res_sat_exp_;
      };
    }  // namespace PAR

    class Anand : public Law
    {
     public:
      using ErrorType = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType;
      using PlasticStrainRateDerivs =
          Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs;

      explicit Anand(Core::Mat::PAR::Parameter* params);

      Mat::Viscoplastic::PAR::Anand* parameter() const override
      {
        return dynamic_cast<Mat::Viscoplastic::PAR::Anand*>(Mat::Viscoplastic::Law::parameter());
      }

      double evaluate_stress_ratio(
          const double equiv_stress, const double equiv_plastic_strain) override;

      double compute_flow_resistance(const double equiv_stress, const double equiv_plastic_strain,
          ErrorType& err_status) override;

      double evaluate_plastic_strain_rate(const double equiv_stress,
          const double equiv_plastic_strain, const double dt, const double max_plastic_strain_incr,
          ErrorType& err_status, const bool update_hist_var = true) override;

      PlasticStrainRateDerivs evaluate_derivatives_of_plastic_strain_rate(const double equiv_stress,
          const double equiv_plastic_strain, const double dt,
          const double max_plastic_strain_deriv_incr, ErrorType& err_status,
          const bool update_hist_var = true) override;

      void setup(const int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

      void pre_evaluate(const Teuchos::ParameterList& params, int gp) override;

      void update() override;

      void update_gp_state(int gp) override;

      void pack_viscoplastic_law(Core::Communication::PackBuffer& data) const override;

      void unpack_viscoplastic_law(Core::Communication::UnpackBuffer& buffer) override;

      std::string debug_get_error_info(int gp);

      void debug_set_last_values(int gp, double last_flow_resistance, double last_plastic_strain);

      void register_output_data_names(
          std::unordered_map<std::string, int>& names_and_size) const override;

      bool evaluate_output_data(
          const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

     private:
      struct ConstPars
      {
        double p;
        double e;
        double s0;
        double H0;
        double a;
        double S_star;
        double n;
        double log_p;
        double log_p_e;
        double eN;
        double inv_S_star;
        double H_0;
        double aH_0;

        ConstPars(const double strain_rate_pre_fac, const double strain_rate_sensitivity,
            const double init_flow_res, const double harden_rate_pre_fac,
            const double harden_rate_sensitivity, const double flow_res_sat_fac,
            const double flow_res_sat_exp)
            : p(strain_rate_pre_fac),
              e(strain_rate_sensitivity),
              s0(init_flow_res),
              H0(harden_rate_pre_fac),
              a(harden_rate_sensitivity),
              S_star(flow_res_sat_fac),
              n(flow_res_sat_exp),
              log_p(std::log(p)),
              log_p_e(std::log(p * e)),
              eN(e * n),
              inv_S_star(1.0 / S_star),
              H_0(H0),
              aH_0(a * H0)
        {
        }
      };

      struct TimeStepQuantities
      {
        std::vector<double> last_flow_resistance_;
        std::vector<double> current_flow_resistance_;
        std::vector<double> last_substep_flow_resistance_;
        std::vector<double> last_plastic_strain_;
        std::vector<double> current_plastic_strain_;
        std::vector<double> last_substep_plastic_strain_;
      };

      double compute_hardening_tangent(
          double equiv_stress, double flow_resistance, ErrorType& err_status);

      Core::LinAlg::Matrix<2, 1> compute_derivatives_of_hardening_tangent(
          double equiv_stress, double flow_resistance, ErrorType& err_status);

      Core::LinAlg::Matrix<2, 1> compute_derivatives_of_flow_resistance(double equiv_stress,
          double flow_resistance, double delta_plastic_strain, ErrorType& err_status);

      const ConstPars const_pars_;
      TimeStepQuantities time_step_quantities_;
    };
  }  // namespace Viscoplastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
