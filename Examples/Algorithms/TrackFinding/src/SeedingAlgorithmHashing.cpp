// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/SeedingAlgorithmHashing.hpp"

#include "Acts/Seeding/BinFinder.hpp"
#include "Acts/Seeding/BinnedSPGroup.hpp"
#include "Acts/Seeding/Seed.hpp"
#include "Acts/Seeding/SeedFilter.hpp"
#include "Acts/Seeding/SeedFinder.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

#include <stdexcept>

ActsExamples::SeedingAlgorithmHashing::SeedingAlgorithmHashing(
    ActsExamples::SeedingAlgorithmHashing::Config cfg, Acts::Logging::Level lvl)
    : ActsExamples::BareAlgorithm("SeedingAlgorithm", lvl),
      m_cfg(std::move(cfg)) {
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument("Missing space point input collections");
  }
  for (const auto& i : m_cfg.inputSpacePoints) {
    if (i.empty()) {
      throw std::invalid_argument("Invalid space point input collection");
    }
  }
  if (m_cfg.outputProtoTracks.empty()) {
    throw std::invalid_argument("Missing proto tracks output collection");
  }
  if (m_cfg.outputSeeds.empty()) {
    throw std::invalid_argument("Missing seeds output collection");
  }

  if (m_cfg.gridConfig.rMax != m_cfg.seedFinderConfig.rMax and
      m_cfg.allowSeparateRMax == false) {
    throw std::invalid_argument(
        "Inconsistent config rMax: using different values in gridConfig and "
        "seedFinderConfig. If values are intentional set allowSeparateRMax to "
        "true");
  }

  if (m_cfg.seedFilterConfig.deltaRMin != m_cfg.seedFinderConfig.deltaRMin) {
    throw std::invalid_argument("Inconsistent config deltaRMin");
  }

  if (m_cfg.gridConfig.deltaRMax != m_cfg.seedFinderConfig.deltaRMax) {
    throw std::invalid_argument("Inconsistent config deltaRMax");
  }

  static_assert(std::numeric_limits<decltype(
                    m_cfg.seedFinderConfig.deltaRMaxTopSP)>::has_quiet_NaN,
                "Value of deltaRMaxTopSP must support NaN values");

  static_assert(std::numeric_limits<decltype(
                    m_cfg.seedFinderConfig.deltaRMinTopSP)>::has_quiet_NaN,
                "Value of deltaRMinTopSP must support NaN values");

  static_assert(std::numeric_limits<decltype(
                    m_cfg.seedFinderConfig.deltaRMaxBottomSP)>::has_quiet_NaN,
                "Value of deltaRMaxBottomSP must support NaN values");

  static_assert(std::numeric_limits<decltype(
                    m_cfg.seedFinderConfig.deltaRMinBottomSP)>::has_quiet_NaN,
                "Value of deltaRMinBottomSP must support NaN values");

  if (std::isnan(m_cfg.seedFinderConfig.deltaRMaxTopSP)) {
    m_cfg.seedFinderConfig.deltaRMaxTopSP = m_cfg.seedFinderConfig.deltaRMax;
  }

  if (std::isnan(m_cfg.seedFinderConfig.deltaRMinTopSP)) {
    m_cfg.seedFinderConfig.deltaRMinTopSP = m_cfg.seedFinderConfig.deltaRMin;
  }

  if (std::isnan(m_cfg.seedFinderConfig.deltaRMaxBottomSP)) {
    m_cfg.seedFinderConfig.deltaRMaxBottomSP = m_cfg.seedFinderConfig.deltaRMax;
  }

  if (std::isnan(m_cfg.seedFinderConfig.deltaRMinBottomSP)) {
    m_cfg.seedFinderConfig.deltaRMinBottomSP = m_cfg.seedFinderConfig.deltaRMin;
  }

  if (m_cfg.gridConfig.zMin != m_cfg.seedFinderConfig.zMin) {
    throw std::invalid_argument("Inconsistent config zMin");
  }

  if (m_cfg.gridConfig.zMax != m_cfg.seedFinderConfig.zMax) {
    throw std::invalid_argument("Inconsistent config zMax");
  }

  if (m_cfg.seedFilterConfig.maxSeedsPerSpM !=
      m_cfg.seedFinderConfig.maxSeedsPerSpM) {
    throw std::invalid_argument("Inconsistent config maxSeedsPerSpM");
  }

  if (m_cfg.gridConfig.cotThetaMax != m_cfg.seedFinderConfig.cotThetaMax) {
    throw std::invalid_argument("Inconsistent config cotThetaMax");
  }

  if (m_cfg.gridConfig.minPt != m_cfg.seedFinderConfig.minPt) {
    throw std::invalid_argument("Inconsistent config minPt");
  }

  if (m_cfg.gridConfig.bFieldInZ != m_cfg.seedFinderOptions.bFieldInZ) {
    throw std::invalid_argument("Inconsistent config bFieldInZ");
  }

  if (m_cfg.gridConfig.zBinEdges.size() - 1 != m_cfg.zBinNeighborsTop.size() &&
      m_cfg.zBinNeighborsTop.empty() == false) {
    throw std::invalid_argument("Inconsistent config zBinNeighborsTop");
  }

  if (m_cfg.gridConfig.zBinEdges.size() - 1 !=
          m_cfg.zBinNeighborsBottom.size() &&
      m_cfg.zBinNeighborsBottom.empty() == false) {
    throw std::invalid_argument("Inconsistent config zBinNeighborsBottom");
  }

  if (!m_cfg.seedFinderConfig.zBinsCustomLooping.empty()) {
    // check if zBinsCustomLooping contains numbers from 1 to the total number
    // of bin in zBinEdges
    for (size_t i = 1; i != m_cfg.gridConfig.zBinEdges.size(); i++) {
      if (std::find(m_cfg.seedFinderConfig.zBinsCustomLooping.begin(),
                    m_cfg.seedFinderConfig.zBinsCustomLooping.end(),
                    i) == m_cfg.seedFinderConfig.zBinsCustomLooping.end()) {
        throw std::invalid_argument(
            "Inconsistent config zBinsCustomLooping does not contain the same "
            "bins as zBinEdges");
      }
    }
  }

  if (m_cfg.seedFinderConfig.useDetailedDoubleMeasurementInfo) {
    m_cfg.seedFinderConfig.getTopHalfStripLength.connect(
        [](const void*, const SimSpacePoint& sp) -> float {
          return sp.topHalfStripLength();
        });

    m_cfg.seedFinderConfig.getBottomHalfStripLength.connect(
        [](const void*, const SimSpacePoint& sp) -> float {
          return sp.bottomHalfStripLength();
        });

    m_cfg.seedFinderConfig.getTopStripDirection.connect(
        [](const void*, const SimSpacePoint& sp) -> Acts::Vector3 {
          return sp.topStripDirection();
        });

    m_cfg.seedFinderConfig.getBottomStripDirection.connect(
        [](const void*, const SimSpacePoint& sp) -> Acts::Vector3 {
          return sp.bottomStripDirection();
        });

    m_cfg.seedFinderConfig.getStripCenterDistance.connect(
        [](const void*, const SimSpacePoint& sp) -> Acts::Vector3 {
          return sp.stripCenterDistance();
        });

    m_cfg.seedFinderConfig.getTopStripCenterPosition.connect(
        [](const void*, const SimSpacePoint& sp) -> Acts::Vector3 {
          return sp.topStripCenterPosition();
        });
  }

  m_cfg.seedFinderConfig.seedFilter =
      std::make_unique<Acts::SeedFilter<SimSpacePoint>>(m_cfg.seedFilterConfig);
}

ActsExamples::ProcessCode ActsExamples::SeedingAlgorithmHashing::execute(
    const AlgorithmContext& ctx) const {
  // seeding looks for nearly straight lines in r/z plane
  // it uses a grid to limit the search to neighbouring grid cells

  // construct the seeding tools
  auto bottomBinFinder = std::make_shared<Acts::BinFinder<SimSpacePoint>>(
      Acts::BinFinder<SimSpacePoint>(m_cfg.zBinNeighborsBottom,
                                    m_cfg.numPhiNeighbors));
  auto topBinFinder = std::make_shared<Acts::BinFinder<SimSpacePoint>>(
      Acts::BinFinder<SimSpacePoint>(m_cfg.zBinNeighborsTop,
                                    m_cfg.numPhiNeighbors));

  auto finder = Acts::SeedFinder<SimSpacePoint>(m_cfg.seedFinderConfig,
                                                m_cfg.seedFinderOptions);

  // covariance tool, extracts covariances per spacepoint as required
  auto extractGlobalQuantities =
      [=](const SimSpacePoint& sp, float, float,
          float) -> std::pair<Acts::Vector3, Acts::Vector2> {
    Acts::Vector3 position{sp.x(), sp.y(), sp.z()};
    Acts::Vector2 covariance{sp.varianceR(), sp.varianceZ()};
    return std::make_pair(position, covariance);
  };

  // construct the combined input container of space point pointers from all
  // configured input sources.
  // pre-compute the total size required so we only need to allocate once
  // size_t nSpacePoints = 0;
  // for (const auto& isp : m_cfg.inputSpacePoints) {
  //  nSpacePoints += ctx.eventStore.get<SimSpacePointContainer>(isp).size();
  // }

  // pre-compute the maximum size required so we only need to allocate once
  // doesn't combine the input containers of space point pointers
  size_t maxNSpacePoints = 0, inSpacePoints = 0;
  for (const auto& isp : m_cfg.inputSpacePoints) {
    inSpacePoints = ctx.eventStore.get<SimSpacePointContainer>(isp).size();
    if (inSpacePoints > maxNSpacePoints){
      maxNSpacePoints = inSpacePoints;
    }
  }

  static thread_local SimSeedContainer seeds;
  seeds.clear();
  static thread_local decltype(finder)::SeedingState state;

  std::vector<const SimSpacePoint*> spacePointPtrs;
  //spacePointPtrs.reserve(nSpacePoints);
  spacePointPtrs.reserve(maxNSpacePoints);
  for (const auto& isp : m_cfg.inputSpacePoints) {
    // construct the seeding tools
    auto grid =
        Acts::SpacePointGridCreator::createGrid<SimSpacePoint>(m_cfg.gridConfig);

    // spacePointPtrs corresponds to a bucket
    spacePointPtrs.clear();

    // is there a way to store the pointers only once?
    for (const auto& spacePoint :
         ctx.eventStore.get<SimSpacePointContainer>(isp)) {
      // since the event store owns the space points, their pointers should be
      // stable and we do not need to create local copies.
      spacePointPtrs.push_back(&spacePoint);
    }

    // extent used to store r range for middle spacepoint
    Acts::Extent rRangeSPExtent;
    // groups spacepoints
    // could be moved before the space point containers loop if is updated dynamically 
    auto spacePointsGrouping = Acts::BinnedSPGroup<SimSpacePoint>(
      spacePointPtrs.begin(), spacePointPtrs.end(), extractGlobalQuantities,
      bottomBinFinder, topBinFinder, std::move(grid), rRangeSPExtent,
      m_cfg.seedFinderConfig, m_cfg.seedFinderOptions);

    // variable middle SP radial region of interest
    const Acts::Range1D<float> rMiddleSPRange(
        std::floor(rRangeSPExtent.min(Acts::binR) / 2) * 2 +
            m_cfg.seedFinderConfig.deltaRMiddleMinSPRange,
        std::floor(rRangeSPExtent.max(Acts::binR) / 2) * 2 -
            m_cfg.seedFinderConfig.deltaRMiddleMaxSPRange);

    // run the seeding
    auto group = spacePointsGrouping.begin(); // could be done before the loop?
    auto groupEnd = spacePointsGrouping.end();
    // if add middle space point -> need to skip compat with previous tops and bottoms
    // [middle, top] done pairs if no new bottom
    // [bottom, middle, top] pairs other wise

    // [middle] not done for new middles
    // [middle, top] of not done pairs for new tops
    // [middle, top, bottom] for new bottoms

    // if add top space point -> need to skip compat with previous middle and bottoms
    // if add bottom space point -> need to skip compat with previous middle and tops
    for (; !(group == groupEnd); ++group) {
      finder.createSeedsForGroup(state, std::back_inserter(seeds), group.bottom(),
                                group.middle(), group.top(), rMiddleSPRange);
    }
  }

  // extract proto tracks, i.e. groups of measurement indices, from tracks seeds
  size_t nSeeds = seeds.size();
  static thread_local ProtoTrackContainer protoTracks;
  protoTracks.clear();

  protoTracks.reserve(nSeeds);
  for (const auto& seed : seeds) {
    ProtoTrack& protoTrack = protoTracks.emplace_back();
    protoTrack.reserve(seed.sp().size());
    for (auto spacePointPtr : seed.sp()) {
      for (const auto slink : spacePointPtr->sourceLinks()) {
        const auto islink = static_cast<const IndexSourceLink&>(*slink);
        protoTrack.emplace_back(islink.index());
      }
    }
  }

  ACTS_DEBUG("Created " << seeds.size() << " track seeds from "
                        << spacePointPtrs.size() << " space points");

  ctx.eventStore.add(m_cfg.outputSeeds, SimSeedContainer{seeds});
  ctx.eventStore.add(m_cfg.outputProtoTracks, ProtoTrackContainer{protoTracks});
  return ActsExamples::ProcessCode::SUCCESS;
}

// ActsExamples::ProcessCode ActsExamples::SeedingAlgorithm::execute(
//     const AlgorithmContext& ctx) const {
//   // construct the combined input container of space point pointers from all
//   // configured input sources.
//   // pre-compute the total size required so we only need to allocate once
//   size_t nSpacePoints = 0;
//   for (const auto& isp : m_cfg.inputSpacePoints) {
//     nSpacePoints += ctx.eventStore.get<SimSpacePointContainer>(isp).size();
//   }

//   std::vector<const SimSpacePoint*> spacePointPtrs;
//   spacePointPtrs.reserve(nSpacePoints);
//   for (const auto& isp : m_cfg.inputSpacePoints) {
//     for (const auto& spacePoint :
//          ctx.eventStore.get<SimSpacePointContainer>(isp)) {
//       // since the event store owns the space points, their pointers should be
//       // stable and we do not need to create local copies.
//       spacePointPtrs.push_back(&spacePoint);
//     }
//   }

//   // construct the seeding tools
//   // covariance tool, extracts covariances per spacepoint as required
//   auto extractGlobalQuantities =
//       [=](const SimSpacePoint& sp, float, float,
//           float) -> std::pair<Acts::Vector3, Acts::Vector2> {
//     Acts::Vector3 position{sp.x(), sp.y(), sp.z()};
//     Acts::Vector2 covariance{sp.varianceR(), sp.varianceZ()};
//     return std::make_pair(position, covariance);
//   };

//   // extent used to store r range for middle spacepoint
//   Acts::Extent rRangeSPExtent;

//   auto bottomBinFinder = std::make_shared<Acts::BinFinder<SimSpacePoint>>(
//       Acts::BinFinder<SimSpacePoint>(m_cfg.zBinNeighborsBottom,
//                                      m_cfg.numPhiNeighbors));
//   auto topBinFinder = std::make_shared<Acts::BinFinder<SimSpacePoint>>(
//       Acts::BinFinder<SimSpacePoint>(m_cfg.zBinNeighborsTop,
//                                      m_cfg.numPhiNeighbors));
//   auto grid =
//       Acts::SpacePointGridCreator::createGrid<SimSpacePoint>(m_cfg.gridConfig);
//   auto spacePointsGrouping = Acts::BinnedSPGroup<SimSpacePoint>(
//       spacePointPtrs.begin(), spacePointPtrs.end(), extractGlobalQuantities,
//       bottomBinFinder, topBinFinder, std::move(grid), rRangeSPExtent,
//       m_cfg.seedFinderConfig, m_cfg.seedFinderOptions);
//   auto finder = Acts::SeedFinder<SimSpacePoint>(m_cfg.seedFinderConfig,
//                                                 m_cfg.seedFinderOptions);

//   // variable middle SP radial region of interest
//   const Acts::Range1D<float> rMiddleSPRange(
//       std::floor(rRangeSPExtent.min(Acts::binR) / 2) * 2 +
//           m_cfg.seedFinderConfig.deltaRMiddleMinSPRange,
//       std::floor(rRangeSPExtent.max(Acts::binR) / 2) * 2 -
//           m_cfg.seedFinderConfig.deltaRMiddleMaxSPRange);

//   // run the seeding
//   static thread_local SimSeedContainer seeds;
//   seeds.clear();
//   static thread_local decltype(finder)::SeedingState state;

//   auto group = spacePointsGrouping.begin();
//   auto groupEnd = spacePointsGrouping.end();
//   for (; !(group == groupEnd); ++group) {
//     finder.createSeedsForGroup(state, std::back_inserter(seeds), group.bottom(),
//                                group.middle(), group.top(), rMiddleSPRange);
//   }

//   // extract proto tracks, i.e. groups of measurement indices, from tracks seeds
//   size_t nSeeds = seeds.size();
//   static thread_local ProtoTrackContainer protoTracks;
//   protoTracks.clear();

//   protoTracks.reserve(nSeeds);
//   for (const auto& seed : seeds) {
//     ProtoTrack& protoTrack = protoTracks.emplace_back();
//     protoTrack.reserve(seed.sp().size());
//     for (auto spacePointPtr : seed.sp()) {
//       for (const auto slink : spacePointPtr->sourceLinks()) {
//         const auto islink = static_cast<const IndexSourceLink&>(*slink);
//         protoTrack.emplace_back(islink.index());
//       }
//     }
//   }

//   ACTS_DEBUG("Created " << seeds.size() << " track seeds from "
//                         << spacePointPtrs.size() << " space points");

//   ctx.eventStore.add(m_cfg.outputSeeds, SimSeedContainer{seeds});
//   ctx.eventStore.add(m_cfg.outputProtoTracks, ProtoTrackContainer{protoTracks});
//   return ActsExamples::ProcessCode::SUCCESS;
// }