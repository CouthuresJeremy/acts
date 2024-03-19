// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// #include "ActsExamples/Framework/IAlgorithm.hpp"
// #include "ActsExamples/Framework/DataHandle.hpp"
// #include "ActsExamples/EventData/SimSpacePoint.hpp"
// #include "ActsExamples/Framework/ProcessCode.hpp"

#include "Acts/Seeding/Hashing/kissrandom.h"
#include "Acts/Seeding/Hashing/annoylib_custom.h"
#include "Acts/Seeding/Hashing/HashingAlgorithmConfig.hpp"

// #include "Acts/Utilities/Logger.hpp"

#include <cstddef>
#include <string>

namespace Acts {

using AnnoyMetric = Annoy::AngularEuclidean;
using AnnoyModel = Annoy::AnnoyIndex<unsigned int, double, AnnoyMetric, Annoy::Kiss32Random, 
                  Annoy::AnnoyIndexSingleThreadedBuildPolicy>;

/// Print hits within some geometric region-of-interest.
template <typename external_spacepoint_t, typename SpacePointContainer>
class HashingAlgorithm {
 public:
  HashingAlgorithm(const HashingAlgorithmConfig& cfg);

  std::vector<SpacePointContainer> execute(SpacePointContainer spacePoints, AnnoyModel annoyModel) const;

  /// Get readonly access to the config parameters
  const Acts::HashingAlgorithmConfig& config() const { return m_cfg; }

 private:
  HashingAlgorithmConfig m_cfg;
};

}  // namespace Acts
