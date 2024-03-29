// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// #include <cstddef>
// #include <string>

namespace Acts {
struct HashingAlgorithmConfig {
  /// Size of the buckets = number of spacepoints in the bucket
  unsigned int bucketSize = 10;

  /// Number of zBins
  unsigned int zBins = 0;
  /// Number of phiBins
  unsigned int phiBins = 50;

  /// Output bucket collection.
  // std::string outputBuckets;
};

}  // namespace Acts
