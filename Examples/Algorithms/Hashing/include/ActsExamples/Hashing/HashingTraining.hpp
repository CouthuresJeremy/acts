// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"

#include "ActsExamples/Hashing/kissrandom.h"
#include "ActsExamples/Hashing/annoylib_custom.h"

#include <cstddef>
#include <string>

namespace ActsExamples {

using AnnoyMetric = Annoy::AngularEuclidean;
using AnnoyModel = Annoy::AnnoyIndex<unsigned int, double, AnnoyMetric, Annoy::Kiss32Random, 
                  Annoy::AnnoyIndexSingleThreadedBuildPolicy>;

/// Print hits within some geometric region-of-interest.
class HashingTrainingAlgorithm final : public IAlgorithm {
// class HashingTrainingAlgorithm {
 public:
  struct Config {
    /// Input space points collection
    std::string inputSpacePoints;
    /**
    /// Input space point collections.
    ///
    /// We allow multiple space point collections to allow different parts of
    /// the detector to use different algorithms for space point construction,
    /// e.g. single-hit space points for pixel-like detectors or double-hit
    /// space points for strip-like detectors.
    std::vector<std::string> inputSpacePoints;
    **/
    /// Size of the buckets = number of hits in the bucket
    unsigned int AnnoySeed = 123456789;

    /// Number of features to use
    int32_t f = 1;
  };

  HashingTrainingAlgorithm(const Config& cfg, Acts::Logging::Level level);

  ProcessCode execute(const AlgorithmContext& ctx) const final override;

  // / Get readonly access to the config parameters
  const Config& config() const { return m_cfg; }

 private:
  Config m_cfg;

  // std::vector<std::unique_ptr<ReadDataHandle<SimSpacePointContainer>>>
  //     m_inputSpacePoints{};
    
  ReadDataHandle<SimSpacePointContainer> m_inputSpacePoints{this, "inputSpacePoints"};

  WriteDataHandle<AnnoyModel> m_outputAnnoyModel{this, "OutputAnnoyModel"};
};

}  // namespace ActsExamples
