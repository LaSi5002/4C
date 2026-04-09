// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CROSSLINKING_HANDLER_HPP
#define FOUR_C_BEAMINTERACTION_CROSSLINKING_HANDLER_HPP

#include "4C_config.hpp"

#include "4C_binstrategy.hpp"
#include "4C_comm_exporter.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::Nodes
{
  class Node;
}

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace Core::LinAlg
{
  class MapExtractor;
}

namespace BeamInteraction
{
  class BeamCrosslinkerHandler
  {
   public:
    // constructor
    BeamCrosslinkerHandler(
        int myrank, std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy);

    // destructor
    virtual ~BeamCrosslinkerHandler() = default;

    /// get binning strategy
    virtual inline std::shared_ptr<Core::Binstrategy::BinningStrategy>& bin_strategy()
    {
      return binstrategy_;
    }

    virtual inline Core::Binstrategy::BinningStrategy const& bin_strategy() const
    {
      return *binstrategy_;
    }

    /// initial distribution of linker ( more general nodes of bindis_ ) to bins
    virtual void distribute_linker_to_bins(std::shared_ptr<Core::LinAlg::Map> const& linkerrowmap);

    /// remove all linker
    virtual void remove_all_linker();

    /// get bin colume map
    virtual inline std::shared_ptr<Core::LinAlg::Map>& bin_col_map() { return bincolmap_; }

    /// get myrank
    virtual inline int my_rank() { return myrank_; }

    /// linker are checked whether they have moved out of their current bin
    /// and transferred if necessary
    virtual std::shared_ptr<std::list<int>> transfer_linker(bool const fill_using_ghosting = true);

    /// node is placed into the correct row bin or put into the list of homeless linker
    virtual bool place_node_correctly(
        std::shared_ptr<Core::Nodes::Node> node,  ///< node to be placed
        const double* currpos,                    ///< current position of this node
        std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker  ///< list of homeless linker
    );

    /// round robin loop to fill linker into its correct bin on according proc
    virtual void fill_linker_into_bins_round_robin(
        std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker  ///< list of homeless linker
    );

    /// get neighbouring bins of linker containing boundary row bins
    virtual void get_neighbouring_bins_of_linker_containing_boundary_row_bins(
        std::set<int>& colbins) const;

   protected:
    /// fill linker into their correct bin on according proc using remote id list
    virtual std::shared_ptr<std::list<int>> fill_linker_into_bins_remote_id_list(
        std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker  ///< set of homeless linker
    );

    /// fill linker into their correct bin on according proc using one layer ghosting
    /// note, this is faster than the other two method as there is no communication required
    /// to find new owner ( complete one layer bin ghosting is required though)
    virtual std::shared_ptr<std::list<int>> fill_linker_into_bins_using_ghosting(
        std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker  ///< set of homeless linker
    );

    /// receive linker and fill them in correct bin
    virtual void receive_linker_and_fill_them_in_bins(int const numrec,
        Core::Communication::Exporter& exporter,
        std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker);

   private:
    /// binning strategy
    std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy_;

    /// myrank
    int myrank_;

    /// colmap of bins
    std::shared_ptr<Core::LinAlg::Map> bincolmap_;
  };

}  // namespace BeamInteraction


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
