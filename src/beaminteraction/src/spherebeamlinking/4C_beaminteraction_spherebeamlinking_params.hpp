// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_SPHEREBEAMLINKING_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_SPHEREBEAMLINKING_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_crosslinking_submodel_evaluator.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN



// forward declaration

namespace Solid
{
  namespace TimeInt
  {
    class BaseDataGlobalState;
  }
}  // namespace Solid
namespace Mat
{
  class CrosslinkerMat;
}
namespace BeamInteraction
{
  /*!
   * data container for input file parameters for submodel crosslinking in beam interaction */
  class SphereBeamLinkingParams
  {
   public:
    //! constructor
    explicit SphereBeamLinkingParams(Solid::TimeInt::BaseDataGlobalState const& gstate);

    //! destructor
    virtual ~SphereBeamLinkingParams() = default;

    //! reset time step in case structure time is adapted during simulation time
    void reset_time_step(double structure_delta_time);

    /// linker material id
    std::shared_ptr<Mat::CrosslinkerMat> get_linker_material() const
    {
      /// HACK: FIX IF MORE THAN ONE CROSSLINKER TYPE
      return mat_.back();
    };

    /// time step for stochastic events concerning crosslinking
    double const& delta_time() const
    {
      return deltatime_;
    };

    /// contraction rate of cell (integrin linker) in [microm/s]
    double contraction_rate(BeamInteraction::CrosslinkerType linkertype) const
    {
      return contractionrate_.at(linkertype);
    };

    /// number of linker per type
    std::vector<int> const& max_num_linker_per_type() const
    {
      return maxnumlinkerpertype_;
    };

    /// material number for linker types
    std::vector<int> const& mat_linker_per_type() const
    {
      return matlinkerpertype_;
    };

    /// get all active linker types
    std::vector<BeamInteraction::CrosslinkerType> const& linker_types() const
    {
      return linkertypes_;
    };

    // distance between two binding spots on a filament
    double filament_bspot_interval_global(BeamInteraction::CrosslinkerType linkertype) const
    {
      return filamentbspotintervalglobal_.at(linkertype);
    };

    // distance between two binding spots on a filament
    double filament_bspot_interval_local(BeamInteraction::CrosslinkerType linkertype) const
    {
      return filamentbspotintervallocal_.at(linkertype);
    };

    // start and end arc parameter for binding spots on a filament
    std::pair<double, double> const& filament_bspot_range_local(
        BeamInteraction::CrosslinkerType linkertype) const
    {
      return filamentbspotrangelocal_.at(linkertype);
    };

    // start and end arc parameter for binding spots on a filament
    std::pair<double, double> const& filament_bspot_range_global(
        BeamInteraction::CrosslinkerType linkertype) const
    {
      return filamentbspotrangeglobal_.at(linkertype);
    };

   private:
    /// time step for stochastic events concerning integrins, e.g. catch-slip-bond behavior
    double deltatime_;
    bool own_deltatime_;
    /// contraction rate of cell (integrin linker) in [microm/s]
    std::map<BeamInteraction::CrosslinkerType, double> contractionrate_;
    /// crosslinker material
    std::vector<std::shared_ptr<Mat::CrosslinkerMat>> mat_;
    /// number of crosslinkers in the simulated volume
    std::vector<int> maxnumlinkerpertype_;
    /// material numbers for crosslinker types
    std::vector<int> matlinkerpertype_;
    /// linker and therefore binding spot types
    std::vector<BeamInteraction::CrosslinkerType> linkertypes_;
    /// distance between two binding spots on each filament
    std::map<BeamInteraction::CrosslinkerType, double> filamentbspotintervalglobal_;
    /// distance between two binding spots on a filament as percentage of filament reference length
    std::map<BeamInteraction::CrosslinkerType, double> filamentbspotintervallocal_;
    /// start and end arc parameter for binding spots on a filament
    std::map<BeamInteraction::CrosslinkerType, std::pair<double, double>> filamentbspotrangeglobal_;
    /// start and end arc parameter for binding spots on a filament
    /// in percent of filament reference length
    std::map<BeamInteraction::CrosslinkerType, std::pair<double, double>> filamentbspotrangelocal_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
