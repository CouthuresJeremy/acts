// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Io/Csv/CsvBucketWriter.hpp"

#include "Acts/Definitions/Units.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/Utilities/Paths.hpp"

#include <ios>
#include <optional>
#include <stdexcept>

#include <dfe/dfe_io_dsv.hpp>

#include "CsvOutputData.hpp"

ActsExamples::CsvBucketWriter::CsvBucketWriter(
    const ActsExamples::CsvBucketWriter::Config& config,
    Acts::Logging::Level level)
    : WriterT(config.inputBuckets, "CsvBucketWriter", level),
      m_cfg(config) {}

ActsExamples::CsvBucketWriter::~CsvBucketWriter() = default;

ActsExamples::ProcessCode ActsExamples::CsvBucketWriter::finalize() {
  // Write the tree
  return ProcessCode::SUCCESS;
}

ActsExamples::ProcessCode ActsExamples::CsvBucketWriter::writeT(
    //const AlgorithmContext& ctx, const SimSpacePointContainer& bucket) {
    const AlgorithmContext& ctx, const std::vector<SimSpacePointContainer>& buckets) {
  // Open per-event file for all components
  std::string pathBucket =
      perEventFilepath(m_cfg.outputDir, "buckets.csv", ctx.eventNumber);

  dfe::NamedTupleCsvWriter<BucketData> writerBucket(pathBucket,
                                                    m_cfg.outputPrecision);

  BucketData bucketData;
  /*
  for (unsigned int bucketIdx = 0; bucketIdx < m_cfg.nBucketsLimit; bucketIdx++){
    //std::vector<int> bucket_ids = AnnoyHashingInstance->m_bucketsSPMap.find(bucketIdx)->second;
    //ctx.eventStore.add("hashingBucket_"+std::to_string(bucketIdx), std::move(bucket_ids));
    SimSpacePointContainer bucket;
    if (bucketIdx < nSpacePoints){
      bucket = AnnoyHashingInstance->m_bucketsSPMap.find(bucketIdx)->second;
    }
    
    //SimHitContainer simHitsBucket;
    //simHitsBucket.insert(bucket.begin(), bucket.end());
    ctx.eventStore.add("hashingSPBucket_"+std::to_string(bucketIdx), std::move(bucket));
  }
  */
  //for (const auto& bucket : buckets) {
  //if (bucket.empty()){
  //continue;
  //}
  /*
    bucketData.measurement_id[0] = (static_cast<const IndexSourceLink&>(*(bucket[0]).sourceLinks()[0])).index();
    bucketData.measurement_id[1] = (static_cast<const IndexSourceLink&>(*(bucket[1]).sourceLinks()[0])).index();
    bucketData.measurement_id[2] = (static_cast<const IndexSourceLink&>(*(bucket[2]).sourceLinks()[0])).index();
    bucketData.measurement_id[3] = (static_cast<const IndexSourceLink&>(*(bucket[3]).sourceLinks()[0])).index();
    bucketData.measurement_id[4] = (static_cast<const IndexSourceLink&>(*(bucket[4]).sourceLinks()[0])).index();
    bucketData.measurement_id[5] = (static_cast<const IndexSourceLink&>(*(bucket[5]).sourceLinks()[0])).index();
    bucketData.measurement_id[6] = (static_cast<const IndexSourceLink&>(*(bucket[6]).sourceLinks()[0])).index();
    bucketData.measurement_id[7] = (static_cast<const IndexSourceLink&>(*(bucket[7]).sourceLinks()[0])).index();
    bucketData.measurement_id[8] = (static_cast<const IndexSourceLink&>(*(bucket[8]).sourceLinks()[0])).index();
    bucketData.measurement_id[9] = (static_cast<const IndexSourceLink&>(*(bucket[9]).sourceLinks()[0])).index();
    bucketData.measurement_id[10] = (static_cast<const IndexSourceLink&>(*(bucket[10]).sourceLinks()[0])).index();
    bucketData.measurement_id[11] = (static_cast<const IndexSourceLink&>(*(bucket[11]).sourceLinks()[0])).index();
    bucketData.measurement_id[12] = (static_cast<const IndexSourceLink&>(*(bucket[12]).sourceLinks()[0])).index();
    bucketData.measurement_id[13] = (static_cast<const IndexSourceLink&>(*(bucket[13]).sourceLinks()[0])).index();
    bucketData.measurement_id[14] = (static_cast<const IndexSourceLink&>(*(bucket[14]).sourceLinks()[0])).index();
    bucketData.measurement_id[15] = (static_cast<const IndexSourceLink&>(*(bucket[15]).sourceLinks()[0])).index();
    bucketData.measurement_id[16] = (static_cast<const IndexSourceLink&>(*(bucket[16]).sourceLinks()[0])).index();
    bucketData.measurement_id[17] = (static_cast<const IndexSourceLink&>(*(bucket[17]).sourceLinks()[0])).index();
    bucketData.measurement_id[18] = (static_cast<const IndexSourceLink&>(*(bucket[18]).sourceLinks()[0])).index();
    bucketData.measurement_id[19] = (static_cast<const IndexSourceLink&>(*(bucket[19]).sourceLinks()[0])).index();
    */
  //for (int SPIdx = 0; SPIdx < bucket.size(); SPIdx++){
  //for (int SPIdx = 0; SPIdx < 20; SPIdx++){
  //  bucketData.measurement_id[SPIdx] = (static_cast<const IndexSourceLink&>(*(bucket[SPIdx]).sourceLinks()[0])).index();
  //  }
  //  writerBucket.append(bucketData);
    //break;
   //}
  
  int bucketIdx = 0;
  for (const auto& bucket : buckets) {
    if (bucket.empty()) {
      continue;
    }
    // for (int SPIdx = 0; SPIdx < bucket.size(); SPIdx++){
    for (int nLines = 0;
         nLines < bucket.size() / 20 + (int)(bucket.size() % 20 != 0);
         nLines++) {
      bucketData.bucketIdx = bucketIdx;
      bucketData.bucketSize = bucket.size();
      for (int SPIdx = 0; SPIdx < 20; SPIdx++) {
        if (nLines * 20 + SPIdx >= bucket.size()) {
          break;
        }
        // bucketData.measurement_id[SPIdx] =
        //     (static_cast<const IndexSourceLink&>(
        //          *(bucket[nLines * 20 + SPIdx]).sourceLinks()[0]))
        //         .index();
        bucketData.measurement_id[SPIdx] =
                 (bucket[nLines * 20 + SPIdx]).sourceLinks()[0].get<IndexSourceLink>()
                .index();
      }
      writerBucket.append(bucketData);
    }
    // break;
    bucketIdx++;
  }

  return ActsExamples::ProcessCode::SUCCESS;
}
