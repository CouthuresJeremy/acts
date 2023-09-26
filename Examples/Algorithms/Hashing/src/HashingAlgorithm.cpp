// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Hashing/HashingAnnoy.hpp"
#include "ActsExamples/Hashing/HashingAlgorithm.hpp"
#include "ActsExamples/Hashing/HashingTraining.hpp"

#include "Acts/Digitization/PlanarModuleCluster.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/GeometryContainers.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/SimParticle.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Utilities/Range.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"

#include "ActsExamples/Hashing/kissrandom.h"
#include "ActsExamples/Hashing/annoylib_custom.h"

#include <vector>


ActsExamples::HashingAlgorithm::HashingAlgorithm(
    const ActsExamples::HashingAlgorithm::Config& cfg, Acts::Logging::Level level)
    : IAlgorithm("HashingAlgorithm", level), m_cfg(cfg) {
  if (m_cfg.bucketSize <= 0) {
    throw std::invalid_argument("Invalid bucket size");
  }
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument("Missing space point input collections");
  }

  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);

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
  if (m_cfg.outputBuckets.empty()) {
    throw std::invalid_argument("Missing buckets output collection");
  }

  m_outputBuckets.initialize(m_cfg.outputBuckets);
}

ActsExamples::ProcessCode ActsExamples::HashingAlgorithm::execute(
    const ActsExamples::AlgorithmContext& ctx) const {

  // const auto& spacePoints =
  //     ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);
  const auto& spacePoints = m_inputSpacePoints(ctx);

  const size_t nSpacePoints = spacePoints.size();

  const unsigned int bucketSize = m_cfg.bucketSize;
  const unsigned int zBins = m_cfg.zBins;
  const unsigned int phiBins = m_cfg.phiBins;

  ACTS_DEBUG("event " << ctx.eventNumber);

  //using AnnoyMetric = Annoy::Angular;
  //using AnnoyMetric = Annoy::Euclidean;
  using AnnoyMetric = Annoy::AngularEuclidean;
  //-DANNOYLIB_MULTITHREADED_BUILD
  //Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy> annoyModel = 
  //Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>(f);

  using AnnoyModel = Annoy::AnnoyIndex<unsigned int, double, AnnoyMetric, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy>;

  // const auto& annoyModel = ctx.eventStore.get<AnnoyModel>("annoyModel");
  const auto& annoyModel = m_inputAnnoyModel(ctx);

  ACTS_DEBUG("annoyModel loaded seed:" << annoyModel.get_seed());
  ACTS_DEBUG("bucketSize:" << bucketSize);
  ACTS_DEBUG("zBins:" << zBins);
  ACTS_DEBUG("phiBins:" << phiBins);

  auto AnnoyHashingInstance = new HashingAnnoy();
  AnnoyHashingInstance->ComputeSpacePointsBuckets(ctx, &annoyModel, spacePoints, bucketSize, zBins, phiBins);

  ACTS_DEBUG("Loaded " << nSpacePoints << " Space Points");

  std::map<int, std::set<ActsExamples::SimSpacePoint>> bucketsSPMap = AnnoyHashingInstance->m_bucketsSPMap;
  std::vector<SimSpacePointContainer> buckets;
  unsigned int nBuckets = (unsigned int)bucketsSPMap.size();
  ACTS_DEBUG("n_buckets:" << nBuckets);
  if (nBuckets > nSpacePoints){
    ACTS_ERROR("More buckets than the number of Space Points");
    assert(false);
  }
  for (unsigned int bucketIdx = 0; bucketIdx < nBuckets; bucketIdx++){
    std::map<int, std::set<ActsExamples::SimSpacePoint>>::iterator iterator=bucketsSPMap.find(bucketIdx);
    if (iterator == bucketsSPMap.end()){
      ACTS_ERROR("Not every bucket have been found");
      assert(false);
    }
    std::set<ActsExamples::SimSpacePoint> bucketSet = iterator->second;
    SimSpacePointContainer bucket;
    for (const auto& spacePoint : bucketSet) {
      bucket.push_back(spacePoint);
    }
    buckets.push_back(bucket);
  }
  m_outputBuckets(ctx, std::move(buckets));

  return ProcessCode::SUCCESS;
}
