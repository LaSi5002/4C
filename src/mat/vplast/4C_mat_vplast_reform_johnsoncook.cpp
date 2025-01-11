// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_vplast_reform_johnsoncook.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

FOUR_C_NAMESPACE_OPEN


using namespace Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::PAR::ReformulatedJohnsonCook::ReformulatedJohnsonCook(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      strain_rate_prefac_(matdata.parameters.get<double>("STRAIN_RATE_PREFAC")),
      strain_rate_exp_fac_(matdata.parameters.get<double>("STRAIN_RATE_EXP_FAC")),
      init_yield_strength_(matdata.parameters.get<double>("INIT_YIELD_STRENGTH")),
      isotrop_harden_prefac_(matdata.parameters.get<double>("ISOTROP_HARDEN_PREFAC")),
      isotrop_harden_exp_(matdata.parameters.get<double>("ISOTROP_HARDEN_EXP")),
      ref_temperature_(matdata.parameters.get<double>("REF_TEMPERATURE")),
      melt_temperature_(matdata.parameters.get<double>("MELT_TEMPERATURE")),
      temperature_sens_(matdata.parameters.get<double>("TEMPERATURE_SENS"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::ReformulatedJohnsonCook::ReformulatedJohnsonCook(
    Core::Mat::PAR::Parameter* params)
    : Mat::Viscoplastic::Law(params),
      const_pars_(parameter()->strain_rate_pre_fac(), 1.0 / parameter()->strain_rate_exp_fac(),
          parameter()->isotrop_harden_prefac(), parameter()->isotrop_harden_exp(),
          parameter()->init_yield_strength())
{
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::Viscoplastic::ReformulatedJohnsonCook::pre_evaluate(
    const Teuchos::ParameterList& params, const int gp, const int ele_gid)
{
  // call pre_evaluate of base class
  Mat::Viscoplastic::Law::pre_evaluate(params, gp, ele_gid);


  // get temperature factors
  double T = parameter()->ref_temperature();
  if (params.isParameter("temperature"))
  {
    T = params.get<double>("temperature");
  }
  const double T_ref = parameter()->ref_temperature();
  const double T_melt = parameter()->melt_temperature();
  const double M = parameter()->temperature_sens();

  // set temperature ratio
  temperature_ratio_ = 1.0;
  if (T != T_ref)
  {
    FOUR_C_ASSERT(T_ref != T_melt,
        "You specified the reference temperature = melting temperature: {}! This cannot be "
        "currently resolved by the Reformulated Johnson-Cook law!",
        T_melt);
    temperature_ratio_ =
        1.0 - (std::pow(T, M) - std::pow(T_ref, M)) / (std::pow(T_melt, M) - std::pow(T_ref, M));
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_stress_ratio(
    const double equiv_stress, const double equiv_plastic_strain)
{
  // extract yield strength from the plastic strain and the material parameters
  const double yield_strength =
      (parameter()->init_yield_strength() * temperature_ratio_ +
          parameter()->isotrop_harden_prefac() * temperature_ratio_ *
              std::pow(equiv_plastic_strain, parameter()->isotrop_harden_exp()));


  return equiv_stress / yield_strength;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_plastic_strain_rate(
    const double equiv_stress, const double equiv_plastic_strain, const double dt,
    const double max_plastic_strain_incr, ErrorType& err_status, const bool update_hist_var)
{
  // first set error status to "no errors"
  err_status = ErrorType::no_errors;

  // Check if plastic strain is negative and throw error (handled by the parent material,
  // substepping)
  if (equiv_plastic_strain < 0.0)
  {
    err_status = ErrorType::negative_plastic_strain;
    return -1;
  }

  // compute the viscoplastic strain rate; first we set it to 0
  double equiv_plastic_strain_rate = 0.0;

  // stress ratio
  double stress_ratio = evaluate_stress_ratio(equiv_stress, equiv_plastic_strain);


  if (stress_ratio >= 1.0)
  {
    double inv_stress_ratio = 1.0 / stress_ratio;
    // save the current values
    if (update_hist_var)
    {
      // GP index safeguard
      FOUR_C_ASSERT_ALWAYS(
          static_cast<size_t>(gp_) < time_step_quantities_.current_yield_strength_.size(),
          "The current gp index is {} while the stored number of GP within Reformulated Johnson - "
          "Cook is {} ",
          gp_, time_step_quantities_.current_yield_strength_.size());

      time_step_quantities_.current_yield_strength_[gp_] = equiv_stress * inv_stress_ratio;
    }
  }

  // then we check the yield condition
  if (stress_ratio >= 1.0)
  {
    // compute logarithm \f$ \log (P \exp(E \left[\frac{\overline{\sigma}}{\sigma_{\text{Y}}}
    // - 1.0]) ) \f$
    const double log_temp = const_pars_.log_p + const_pars_.e * (stress_ratio - 1.0);

    // check if characteristic term too large, throw error overflow
    // error if so
    if (std::log(dt) + log_temp > std::log(max_plastic_strain_incr + const_pars_.p * dt))
    {
      err_status = ErrorType::overflow_error;
      return -1;
    }

    equiv_plastic_strain_rate = std::exp(log_temp) - const_pars_.p;
  }

  return equiv_plastic_strain_rate;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<2, 1>
Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_derivatives_of_plastic_strain_rate(
    const double equiv_stress, const double equiv_plastic_strain, const double dt,
    const double max_plastic_strain_deriv_incr, ErrorType& err_status, const bool update_hist_var)
{
  // declare derivatives to be computed
  Core::LinAlg::Matrix<2, 1> equiv_plastic_strain_rate_ders(Core::LinAlg::Initialization::zero);

  // first set error status to "no errors"
  err_status = ErrorType::no_errors;

  // used equivalent plastic strain
  double used_equiv_plastic_strain = equiv_plastic_strain;

  // check whether the plastic strain is less than a set value (singularity in the derivatives
  // below)
  if (std::abs(equiv_plastic_strain) < 1.0e-16)
  {
    used_equiv_plastic_strain = 1.0e-16;
  }

  // Check if plastic strain is negative and throw error (handled by the parent material,
  // substepping)
  if (equiv_plastic_strain < 0.0)
  {
    err_status = ErrorType::negative_plastic_strain;
    return Core::LinAlg::Matrix<2, 1>{Core::LinAlg::Initialization::zero};
  }

  // extraction of the yield strength from the plastic strain and the material parameters
  const double yield_strength =
      const_pars_.sigma_Y0 + const_pars_.B * std::pow(used_equiv_plastic_strain, const_pars_.N);
  const double log_yield_strength = std::log(yield_strength);
  const double inv_yield_strength = 1.0 / yield_strength;


  // logarithms of equivalent stress and plastic strain
  const double log_equiv_stress = std::log(equiv_stress);
  const double log_equiv_plastic_strain = std::log(used_equiv_plastic_strain);

  // logarithm of the time step
  const double log_dt = std::log(dt);

  // computation of derivatives

  // then we check the yield condition
  if (evaluate_stress_ratio(equiv_stress, equiv_plastic_strain) >= 1.0)
  {
    // compute first the logarithms of our derivatives (try to avoid overflow!)
    double log_deriv_sigma = const_pars_.log_p_e +
                             const_pars_.e * (equiv_stress * inv_yield_strength - 1.0) -
                             log_yield_strength;
    // hardening case
    if (const_pars_.B > 0.0)
    {
      const double log_deriv_eps =
          const_pars_.log_p_e + const_pars_.e * (equiv_stress * inv_yield_strength - 1.0) +
          log_equiv_stress - 2.0 * log_yield_strength + const_pars_.log_B_N +
          (const_pars_.N - 1.0) * log_equiv_plastic_strain;
      // check overflow error using these logarithms
      double log_max_plastic_strain_deriv_value = std::log(max_plastic_strain_deriv_incr);
      if ((log_dt + log_deriv_sigma > log_max_plastic_strain_deriv_value) &&
          (log_dt + log_deriv_eps > log_max_plastic_strain_deriv_value))
      {
        err_status = ErrorType::failed_computation_flow_resistance_derivs;
        return Core::LinAlg::Matrix<2, 1>{Core::LinAlg::Initialization::zero};
      }

      // compute the exact derivatives using these logarithms
      equiv_plastic_strain_rate_ders(0, 0) = std::exp(log_deriv_sigma);
      equiv_plastic_strain_rate_ders(1, 0) = -std::exp(log_deriv_eps);
    }
    // perfect plasticity
    else
    {
      // check overflow error using these logarithms
      double log_max_plastic_strain_deriv_value = std::log(max_plastic_strain_deriv_incr);
      if ((log_dt + log_deriv_sigma > log_max_plastic_strain_deriv_value))
      {
        err_status = ErrorType::failed_computation_flow_resistance_derivs;
        return Core::LinAlg::Matrix<2, 1>{Core::LinAlg::Initialization::zero};
      }

      // compute the exact derivatives using these logarithms
      equiv_plastic_strain_rate_ders(0, 0) = std::exp(log_deriv_sigma);
      equiv_plastic_strain_rate_ders(1, 0) = 0.0;
    }
  }

  return equiv_plastic_strain_rate_ders;
}

void Mat::Viscoplastic::ReformulatedJohnsonCook::setup(const int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  time_step_quantities_.current_yield_strength_.resize(numgp, parameter()->init_yield_strength());
}

void Mat::Viscoplastic::ReformulatedJohnsonCook::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["yield_strength"] = 1;
}

bool Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "yield_strength")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_yield_strength_.size());
        ++gp)
    {
      data(gp, 0) = time_step_quantities_.current_yield_strength_[gp];
    }
    return true;
  }

  return false;
}

void Mat::Viscoplastic::ReformulatedJohnsonCook::pack_viscoplastic_law(
    Core::Communication::PackBuffer& data) const
{
  // pack relevant variables
  if (parameter() != nullptr)
  {
    // we need to pack this current value since it also saves the number
    // of Gauss points
    add_to_pack(data, time_step_quantities_.current_yield_strength_);
  }
}

void Mat::Viscoplastic::ReformulatedJohnsonCook::unpack_viscoplastic_law(
    Core::Communication::UnpackBuffer& buffer)
{
  // NOTE: factory method is called during assign_to_source in the unpack method of the
  // multiplicative split framework --> the param class of the viscoplastic law is created, we only
  // need to unpack the history variables
  if (parameter() != nullptr)
  {
    extract_from_pack(buffer, time_step_quantities_.current_yield_strength_);
  }
}


FOUR_C_NAMESPACE_CLOSE
