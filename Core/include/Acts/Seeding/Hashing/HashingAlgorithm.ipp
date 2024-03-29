// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/Hashing/HashingAnnoy.hpp"

#include <vector>

namespace Acts {
// constructor
template <typename external_spacepoint_t, typename SpacePointContainer>
HashingAlgorithm<external_spacepoint_t, SpacePointContainer>::HashingAlgorithm(
    const HashingAlgorithmConfig& cfg)
    : m_cfg(cfg) {
  if (m_cfg.bucketSize <= 0) {
    throw std::invalid_argument("Invalid bucket size");
  }
  // if (m_cfg.inputSpacePoints.empty()) {
  //   throw std::invalid_argument("Missing space point input collections");
  // }

  // if (m_cfg.outputBuckets.empty()) {
  //   throw std::invalid_argument("Missing buckets output collection");
  // }

  // m_outputBuckets.initialize(m_cfg.outputBuckets);
}

// function to create the buckets of spacepoints.
template <typename external_spacepoint_t, typename SpacePointContainer>
void HashingAlgorithm<external_spacepoint_t, SpacePointContainer>::execute(
    SpacePointContainer& spacePoints,
    AnnoyModel* annoyModel,
    GenericBackInserter<SpacePointContainer> outIt) const {

  using map_t = std::map<unsigned int, std::set<external_spacepoint_t>>;

  // ACTS_DEBUG("Start of HashingAlgorithm execute");

  const size_t nSpacePoints = spacePoints.size();

  const unsigned int bucketSize = m_cfg.bucketSize;
  const unsigned int zBins = m_cfg.zBins;
  const unsigned int phiBins = m_cfg.phiBins;

  // ACTS_DEBUG("annoyModel loaded seed:" << annoyModel.get_seed());
  // ACTS_DEBUG("bucketSize:" << bucketSize);
  // ACTS_DEBUG("zBins:" << zBins);
  // ACTS_DEBUG("phiBins:" << phiBins);

  HashingAnnoy<external_spacepoint_t, SpacePointContainer>* AnnoyHashingInstance = new HashingAnnoy<external_spacepoint_t, SpacePointContainer>();
  AnnoyHashingInstance->ComputeSpacePointsBuckets(annoyModel, spacePoints, bucketSize, zBins, phiBins);

  // ACTS_DEBUG("Loaded " << nSpacePoints << " Space Points");

  map_t bucketsSPMap = AnnoyHashingInstance->m_bucketsSPMap;
  unsigned int nBuckets = (unsigned int)bucketsSPMap.size();
  // ACTS_DEBUG("n_buckets:" << nBuckets);
  if (nBuckets > nSpacePoints){
    throw std::runtime_error("More buckets than the number of Space Points");
  }
  for (unsigned int bucketIdx = 0; bucketIdx < nBuckets; bucketIdx++){
    typename map_t::iterator iterator=bucketsSPMap.find(bucketIdx);
    if (iterator == bucketsSPMap.end()){
      throw std::runtime_error("Not every bucket have been found");
    }
    std::set<external_spacepoint_t> bucketSet = iterator->second;
    SpacePointContainer bucket;
    for (external_spacepoint_t spacePoint : bucketSet) {
      bucket.push_back(spacePoint);
    }
    outIt = SpacePointContainer{bucket};
  }
  // std::vector<SpacePointContainer> buckets;
  // m_outputBuckets(ctx, std::move(buckets));

  // ACTS_DEBUG("End of HashingAlgorithm execute");
}

}  // namespace Acts
