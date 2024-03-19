// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// #pragma once

// #include "ActsExamples/Framework/IAlgorithm.hpp"
// #include "ActsExamples/Framework/DataHandle.hpp"
// #include "ActsExamples/EventData/SimSpacePoint.hpp"
// #include "ActsExamples/Framework/ProcessCode.hpp"

// #include "Acts/Seeding/Hashing/kissrandom.h"
// #include "Acts/Seeding/Hashing/annoylib_custom.h"

// #include "Acts/Utilities/Logger.hpp"

// #include <cstddef>
// #include <string>

namespace Acts {

//Forward declaration
// template <typename SpacePointContainer>
// class HashingAlgorithm;

struct HashingAlgorithmConfig {
  /// Input space points collection
  // std::string inputSpacePoints;
  // /// Input space points collections
  // std::vector<std::string> inputSpacePoints;
  /// Size of the buckets = number of hits in the bucket
  unsigned int bucketSize = 10;

  /// Number of zBins
  unsigned int zBins = 0;
  /// Number of phiBins
  unsigned int phiBins = 50;

  /// Output bucket collection.
  // std::string outputBuckets;
};

}  // namespace Acts
