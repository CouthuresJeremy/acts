// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Seeding/BinFinder.hpp"
#include "Acts/Seeding/SeedFilterConfig.hpp"
#include "Acts/Seeding/SeedFinder.hpp"
#include "Acts/Seeding/SeedFilter.hpp"
#include "Acts/Seeding/SeedFinderConfig.hpp"
#include "Acts/Seeding/Hashing/HashingAlgorithm.hpp"
#include "Acts/Seeding/Hashing/HashingTraining.hpp"
#include "Acts/Seeding/SpacePointGrid.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iterator>
#include <vector>
#include <set>

template <typename T>
class SetPolicy : public ContainerPolicy<T> {
    std::set<T>& container;

public:
    SetPolicy(std::set<T>& s) : container(s) {}

    void policyInsert(T value) override {
        container.insert(value);
    }
};

namespace ActsExamples {
struct AlgorithmContext;

/// Construct track seeds from space points.
class SeedingAlgorithmHashing final : public IAlgorithm {
 public:
  struct Config {
    /// Input space point collections.
    ///
    /// We allow multiple space point collections to allow different parts of
    /// the detector to use different algorithms for space point construction,
    /// e.g. single-hit space points for pixel-like detectors or double-hit
    /// space points for strip-like detectors.
    std::vector<std::string> inputSpacePoints;
    /// Output track seed collection.
    std::string outputSeeds;

    Acts::SeedFilterConfig seedFilterConfig;
    Acts::SeedFinderConfig<SimSpacePoint> seedFinderConfig;
    Acts::SpacePointGridConfig gridConfig;
    Acts::SpacePointGridOptions gridOptions;
    Acts::SeedFinderOptions seedFinderOptions;
    Acts::HashingAlgorithmConfig hashingConfig;
    Acts::HashingTrainingAlgorithmConfig hashingTrainingConfig;

    // allow for different values of rMax in gridConfig and seedFinderConfig
    bool allowSeparateRMax = false;

    // vector containing the map of z bins in the top and bottom layers
    std::vector<std::pair<int, int>> zBinNeighborsTop;
    std::vector<std::pair<int, int>> zBinNeighborsBottom;
    // number of phiBin neighbors at each side of the current bin that will be
    // used to search for SPs
    int numPhiNeighbors = 1;
  };

  /// Construct the seeding algorithm.
  ///
  /// @param cfg is the algorithm configuration
  /// @param lvl is the logging level
  SeedingAlgorithmHashing(Config cfg, Acts::Logging::Level lvl);

  /// Run the seeding algorithm.
  ///
  /// @param ctx is the algorithm context with event information
  /// @return a process code indication success or failure
  ProcessCode execute(const AlgorithmContext& ctx) const final;

  /// Const access to the config
  const Config& config() const { return m_cfg; }

 private:
  Acts::SeedFinder<SimSpacePoint> m_seedFinder;
  std::shared_ptr<const Acts::BinFinder<SimSpacePoint>> m_bottomBinFinder;
  std::shared_ptr<const Acts::BinFinder<SimSpacePoint>> m_topBinFinder;
  Config m_cfg;

  std::vector<std::unique_ptr<ReadDataHandle<SimSpacePointContainer>>>
      m_inputSpacePoints{};

  WriteDataHandle<SimSeedContainer> m_outputSeeds{this, "OutputSeeds"};
  Acts::HashingAlgorithm<SimSpacePoint, std::vector<const SimSpacePoint*>> m_Hashing;
  Acts::HashingTrainingAlgorithm<std::vector<const SimSpacePoint*>> m_HashingTraining;
};

}  // namespace ActsExamples
