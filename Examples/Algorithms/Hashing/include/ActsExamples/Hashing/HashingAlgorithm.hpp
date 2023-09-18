// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/Framework/BareAlgorithm.hpp"

#include <cstddef>
#include <string>

namespace ActsExamples {

/// Print hits within some geometric region-of-interest.
class HashingAlgorithm final : public BareAlgorithm {
 public:
  struct Config {
    /// Input cluster collection.
    std::string inputClusters;
    /// Input hit-particles map.
    std::string inputMeasurementParticlesMap;
    /// Input hit id collection
    std::string inputHitIds;
    /// Input simulated hit collection
    std::string inputSimHits;
    /// Input space points collection
    std::string inputSpacePoints;
    /// Size of the buckets = number of hits in the bucket
    unsigned int bucketSize = 10;

    /// Number of zBins
    unsigned int zBins = 0;
    /// Number of phiBins
    unsigned int phiBins = 50;
  };

  HashingAlgorithm(const Config& cfg, Acts::Logging::Level level);

  ProcessCode execute(const AlgorithmContext& ctx) const final override;

  /// Get readonly access to the config parameters
  const Config& config() const { return m_cfg; }

 private:
  Config m_cfg;
};

}  // namespace ActsExamples
