// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/Hashing/HashingAlgorithm.hpp"
#include "Acts/Seeding/Hashing/HashingAnnoy.hpp"
// #include "Acts/Seeding/Hashing/HashingTraining.hpp"

// #include "Acts/Digitization/PlanarModuleCluster.hpp"
// #include "Acts/Utilities/Logger.hpp"
// #include "ActsExamples/EventData/GeometryContainers.hpp"
// #include "ActsExamples/EventData/Index.hpp"
// #include "ActsExamples/EventData/SimParticle.hpp"
// #include "ActsExamples/Framework/WhiteBoard.hpp"
// #include "ActsExamples/Utilities/Range.hpp"
// #include "ActsExamples/EventData/SimSpacePoint.hpp"

#include "Acts/Seeding/Hashing/kissrandom.h"
#include "Acts/Seeding/Hashing/annoylib_custom.h"

#include <vector>


template <typename external_spacepoint_t, typename SpacePointContainer>
Acts::HashingAlgorithm<external_spacepoint_t, SpacePointContainer>::HashingAlgorithm(
    const Acts::HashingAlgorithmConfig& cfg)
    : m_cfg(cfg) {
  if (m_cfg.bucketSize <= 0) {
    throw std::invalid_argument("Invalid bucket size");
  }
  // if (m_cfg.inputSpacePoints.empty()) {
  //   throw std::invalid_argument("Missing space point input collections");
  // }

  // m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);

  // m_inputAnnoyModel.initialize("OutputAnnoyModel");

  // for (const auto& spName : m_cfg.inputSpacePoints) {
  //   if (spName.empty()) {
  //     throw std::invalid_argument("Invalid space point input collection");
  //   }

  //   auto& handle = m_inputSpacePoints.emplace_back(
  //       std::make_unique<ReadDataHandle<SimSpacePointContainer>>(
  //           this,
  //           "InputSpacePoints#" + std::to_string(m_inputSpacePoints.size())));
  //   handle->initialize(spName);
  // }
  // if (m_cfg.outputBuckets.empty()) {
  //   throw std::invalid_argument("Missing buckets output collection");
  // }

  // m_outputBuckets.initialize(m_cfg.outputBuckets);
}

template <typename external_spacepoint_t, typename SpacePointContainer>
std::vector<SpacePointContainer> Acts::HashingAlgorithm<external_spacepoint_t, SpacePointContainer>::execute(
// void Acts::HashingAlgorithm<SpacePointContainer>::execute(
    // const Acts::AlgorithmContext& ctx
    SpacePointContainer spacePoints,
    AnnoyModel annoyModel) const {

  using map_t = std::map<int, std::set<external_spacepoint_t>>;

  // ACTS_DEBUG("Start of HashingAlgorithm execute");

  // const auto& spacePoints =
  //     ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);
  // const auto& spacePoints = m_inputSpacePoints(ctx);

  const size_t nSpacePoints = spacePoints.size();

  const unsigned int bucketSize = m_cfg.bucketSize;
  const unsigned int zBins = m_cfg.zBins;
  const unsigned int phiBins = m_cfg.phiBins;

  // ACTS_DEBUG("event " << ctx.eventNumber);


  // const auto& annoyModel = ctx.eventStore.get<AnnoyModel>("annoyModel");
  // const auto& annoyModel = m_inputAnnoyModel(ctx);

  // ACTS_DEBUG("annoyModel loaded seed:" << annoyModel.get_seed());
  // ACTS_DEBUG("bucketSize:" << bucketSize);
  // ACTS_DEBUG("zBins:" << zBins);
  // ACTS_DEBUG("phiBins:" << phiBins);

  Acts::HashingAnnoy AnnoyHashingInstance = new HashingAnnoy<external_spacepoint_t, SpacePointContainer>();
  AnnoyHashingInstance->ComputeSpacePointsBuckets(&annoyModel, spacePoints, bucketSize, zBins, phiBins);

  // ACTS_DEBUG("Loaded " << nSpacePoints << " Space Points");

  map_t bucketsSPMap = AnnoyHashingInstance->m_bucketsSPMap;
  std::vector<SpacePointContainer> buckets;
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
    for (const auto& spacePoint : bucketSet) {
      bucket.push_back(spacePoint);
    }
    buckets.push_back(bucket);
  }
  // m_outputBuckets(ctx, std::move(buckets));

  // ACTS_DEBUG("End of HashingAlgorithm execute");

  return buckets;  
}
