// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Hashing/HashingAlgorithm.hpp"
#include "ActsExamples/Hashing/HashingAnnoy.hpp"

#include "Acts/Digitization/PlanarModuleCluster.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/GeometryContainers.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/SimParticle.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Utilities/Range.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"

#include <vector>

ActsExamples::HashingAlgorithm::HashingAlgorithm(
    const ActsExamples::HashingAlgorithm::Config& cfg, Acts::Logging::Level level)
    : BareAlgorithm("HashingAlgorithm", level), m_cfg(cfg) {
  /*
  if (m_cfg.inputClusters.empty()) {
    throw std::invalid_argument("Input clusters collection is not configured");
  }
  if (m_cfg.inputMeasurementParticlesMap.empty()) {
    throw std::invalid_argument(
        "Input hit-particles map collection is not configured");
  }
  if (m_cfg.inputHitIds.empty()) {
    throw std::invalid_argument("Input hit ids collection is not configured");
  }
  */
  if (m_cfg.inputSimHits.empty()) {
    throw std::invalid_argument("Missing input simulated hits collection");
  }
  if (m_cfg.bucketSize <= 0) {
    throw std::invalid_argument("Invalid bucket size");
  }
  if (m_cfg.nBucketsLimit <= 0) {
    throw std::invalid_argument("Invalid superior limit");
  }
}

ActsExamples::ProcessCode ActsExamples::HashingAlgorithm::execute(
    const ActsExamples::AlgorithmContext& ctx) const {
  using Clusters = ActsExamples::GeometryIdMultimap<Acts::PlanarModuleCluster>;
  using HitParticlesMap = ActsExamples::IndexMultimap<ActsFatras::Barcode>;
  using HitIds = std::vector<size_t>;

  /**
  const auto& clusters = ctx.eventStore.get<Clusters>(m_cfg.inputClusters);
  const auto& hitParticlesMap =
      ctx.eventStore.get<HitParticlesMap>(m_cfg.inputMeasurementParticlesMap);
  const auto& hitIds = ctx.eventStore.get<HitIds>(m_cfg.inputHitIds);
  **/
  const auto& simHits =
      ctx.eventStore.get<SimHitContainer>(m_cfg.inputSimHits);

  const auto& spacePoints =
      ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  /**
  size_t nSpacePoints = 0;
  for (const auto& isp : m_cfg.inputSpacePoints) {
    nSpacePoints += spacePoints(isp).size();
  }
  //**/
  size_t nSpacePoints = spacePoints.size();

  const unsigned int bucketSize = m_cfg.bucketSize;
  
  // construct the combined input container of space point pointers from all
  // configured input sources.
  // pre-compute the total size required so we only need to allocate once
  /*
  size_t nSpacePoints = 0;
  for (const auto& isp : m_cfg.inputSpacePoints) {
    nSpacePoints += ctx.eventStore.get<SimSpacePointContainer>(isp).size();
  }
  */

  // extent used to store r range for middle spacepoint
  /*
  Acts::Extent rRangeSPExtent;

  std::vector<const SimSpacePoint*> spacePointPtrs;
  spacePointPtrs.reserve(nSpacePoints);
  for (const auto& isp : m_cfg.inputSpacePoints) {
    for (const auto& spacePoint :
         ctx.eventStore.get<SimSpacePointContainer>(isp)) {
      // since the event store owns the space points, their pointers should be
      // stable and we do not need to create local copies.
      spacePointPtrs.push_back(&spacePoint);
      // store x,y,z values in extent
      rRangeSPExtent.extend({spacePoint.x(), spacePoint.y(), spacePoint.z()});
    }
  }
  */

  /**
  if (clusters.size() != hitIds.size()) {
    ACTS_ERROR(
        "event "
        << ctx.eventNumber
        << " input clusters and hit ids collections have inconsistent size");
    return ProcessCode::ABORT;
  }
  ACTS_INFO("event " << ctx.eventNumber << " collection '"
                     << m_cfg.inputClusters << "' contains " << clusters.size()
                     << " hits");
  **/
  ACTS_INFO("event " << ctx.eventNumber);

  auto AnnoyHashingInstance = new HashingAnnoy();
  //AnnoyHashingInstance->ComputeHitBuckets(ctx, simHits, bucketSize);
  AnnoyHashingInstance->ComputeSpacePointsBuckets(ctx, spacePoints, bucketSize);
  //ACTS_INFO("hit " << AnnoyHashingInstance->m_bucketsMap.find(0)->second);
  //ctx.eventStore.add("bucketsMap", std::move(AnnoyHashingInstance->m_bucketsMap));
  //ACTS_INFO("Loaded " << simHits.size() << " sim hits");
  ACTS_INFO("Loaded " << nSpacePoints << " Space Points");
  //for (const auto& hit : hits) {
  /*
  for (unsigned int bucketIdx = 0; bucketIdx < simHits.size(); bucketIdx++){
    //std::vector<int> bucket_ids = AnnoyHashingInstance->m_bucketsMap.find(bucketIdx)->second;
    //ctx.eventStore.add("hashingBucket_"+std::to_string(bucketIdx), std::move(bucket_ids));
    SimHitContainer::sequence_type bucket = AnnoyHashingInstance->m_bucketsMap.find(bucketIdx)->second;
    SimHitContainer simHitsBucket;
    simHitsBucket.insert(bucket.begin(), bucket.end());
    ctx.eventStore.add("hashingBucket_"+std::to_string(bucketIdx), std::move(simHitsBucket));
  }
  */
  
  /*
  for (unsigned int bucketIdx = 0; bucketIdx < nSpacePoints; bucketIdx++){
    //std::vector<int> bucket_ids = AnnoyHashingInstance->m_bucketsSPMap.find(bucketIdx)->second;
    //ctx.eventStore.add("hashingBucket_"+std::to_string(bucketIdx), std::move(bucket_ids));
    SimSpacePointContainer bucket = AnnoyHashingInstance->m_bucketsSPMap.find(bucketIdx)->second;
    //SimHitContainer simHitsBucket;
    //simHitsBucket.insert(bucket.begin(), bucket.end());
    ctx.eventStore.add("hashingSPBucket_"+std::to_string(bucketIdx), std::move(bucket));
  }
  */
  std::vector<SimSpacePointContainer> buckets;
  for (unsigned int bucketIdx = 0; bucketIdx < m_cfg.nBucketsLimit; bucketIdx++){
    //std::vector<int> bucket_ids = AnnoyHashingInstance->m_bucketsSPMap.find(bucketIdx)->second;
    //ctx.eventStore.add("hashingBucket_"+std::to_string(bucketIdx), std::move(bucket_ids));
    SimSpacePointContainer bucket;
    SimSpacePointContainer bucket2;
    if (bucketIdx < nSpacePoints){
      if (AnnoyHashingInstance->m_bucketsSPMap.find(bucketIdx) != AnnoyHashingInstance->m_bucketsSPMap.end()){
        bucket = AnnoyHashingInstance->m_bucketsSPMap.find(bucketIdx)->second;
        bucket2 = AnnoyHashingInstance->m_bucketsSPMap.find(bucketIdx)->second;
      }
    }
    
    //SimHitContainer simHitsBucket;
    //simHitsBucket.insert(bucket.begin(), bucket.end());
    ctx.eventStore.add("hashingSPBucket_"+std::to_string(bucketIdx), std::move(bucket));
    buckets.push_back(bucket2);
  }
  ctx.eventStore.add("buckets", std::move(buckets));


  /// Split measurementSimHitsMap, sourceLinks, measurementParticlesMap, clusters and measurements
  /// according to bucket
  // can make it faster by comparing simHitIdx to the max(bucketSimHitIdx) and min(bucketSimHitIdx)
  // or by hash through a map
  // or by encoding bucket hits on a boolean of size simHits.size() and compare
  // as a bucket is made for every hit, it might be worth reversing measurementSimHitsMap
  /**
  for (auto sourceLink : sourceLinks){
    // https://github.com/acts-project/acts/blob/main/Examples/Framework/include/ActsExamples/EventData/IndexSourceLink.hpp

    //sourceLink.geometryId();
    Index measurementIdx = sourceLink.index();
    const auto simHitIdx = measurementSimHitsMap.find(measurementIdx);
    for (auto hit: simHitsBucket){
      const auto bucketSimHitIdx = simHits.find(measurementIdx);

    if (simHitIdx == )
    }
  }
  **/

  return ProcessCode::SUCCESS;
}
