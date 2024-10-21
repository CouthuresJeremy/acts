// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"

#include <map>
#include <set>

#include <annoy/annoylib.h>
#include <annoy/kissrandom.h>

namespace Acts {

template <typename external_spacepoint_t, typename SpacePointContainer>
class HashingAnnoy {
 public:
  void ComputeSpacePointsBuckets(
      const Annoy::AnnoyIndex<
          unsigned int, double, Annoy::AngularEuclidean, Annoy::Kiss32Random,
          Annoy::AnnoyIndexSingleThreadedBuildPolicy>* annoyModel,
      const SpacePointContainer& spacePoints, const unsigned int bucketSize,
      const unsigned int zBins, const unsigned int phiBins,
      const double layerRMin, const double layerRMax, const double layerZMin,
      const double layerZMax);
  std::map<unsigned int, std::set<external_spacepoint_t>> m_bucketsSPMap;
};
}  // namespace Acts

namespace Acts::detail {
inline bool LayerSelection(double layerRMin, double layerRMax, double layerZMin,
                           double layerZMax, double r2, double z) {
  bool isInside = (r2 > layerRMin * layerRMin && r2 < layerRMax * layerRMax) &&
                  (z > layerZMin && z < layerZMax);
  return isInside;
}

inline int GetBinIndex(double layerZMin, double layerZMax, double z,
                       unsigned int zBins) {
  using Scalar = Acts::ActsScalar;
  Scalar binSize = (layerZMax - layerZMin) / zBins;
  int binIndex = (int)((z - layerZMin + 0.5 * binSize) / binSize);
  return binIndex;
}

inline int GetBinIndexPhi(double phi, unsigned int phiBins) {
  using Scalar = Acts::ActsScalar;
  Scalar binSize = 2 * M_PI / phiBins;
  int binIndex = (int)((phi + M_PI) / binSize);
  return binIndex;
}
}  // namespace Acts::detail

#include "Acts/Seeding/Hashing/HashingAnnoy.ipp"
