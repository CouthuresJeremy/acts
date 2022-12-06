// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Hashing/HashingAnnoy.hpp"

#include "Acts/Definitions/Units.hpp"
#include "ActsExamples/Hashing/kissrandom.h"
#include "ActsExamples/Hashing/annoylib_custom.h"

#include <map>
#include <vector>


ActsExamples::ProcessCode ActsExamples::HashingAnnoy::ComputeHitBuckets(
    const AlgorithmContext& ctx, 
    const ActsExamples::SimHitContainer& hits,
    const unsigned int bucketSize) {
  using Scalar = Acts::ActsScalar;
  //if (not m_outputFile) {
  //  ACTS_ERROR("Missing output file");
  //  return ProcessCode::ABORT;
  //}

  // ensure exclusive access to tree/file while writing
  //std::lock_guard<std::mutex> lock(m_writeMutex);

  int32_t f = 2; // add it as arg; corresponds to the number of features
  int hitIndex = 0;
  //-DANNOYLIB_MULTITHREADED_BUILD
  //Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy> t = 
  //Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>(f);

  Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy> t = 
  Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy>(f);


  // Get the event number
  int32_t eventId = ctx.eventNumber;
  
  // Add hit parameters to Annoy
  for (const auto& hit : hits) {
    //int32_t particleId = hit.particleId().value();
    //int32_t geometryId = hit.geometryId().value();
    // write hit position
    Scalar tx = hit.fourPosition().x() / Acts::UnitConstants::mm;
    Scalar ty = hit.fourPosition().y() / Acts::UnitConstants::mm;
    Scalar tz = hit.fourPosition().z() / Acts::UnitConstants::mm;
    Scalar tt = hit.fourPosition().w() / Acts::UnitConstants::ns;
    // write four-momentum before interaction
    //Scalar tpx = hit.momentum4Before().x() / Acts::UnitConstants::GeV;
    //Scalar tpy = hit.momentum4Before().y() / Acts::UnitConstants::GeV;
    //Scalar tpz = hit.momentum4Before().z() / Acts::UnitConstants::GeV;
    //Scalar te = hit.momentum4Before().w() / Acts::UnitConstants::GeV;
    // write four-momentum change due to interaction
    //const auto delta4 = hit.momentum4After() - hit.momentum4Before();
    //Scalar deltapx = delta4.x() / Acts::UnitConstants::GeV;
    //Scalar deltapy = delta4.y() / Acts::UnitConstants::GeV;
    //Scalar deltapz = delta4.z() / Acts::UnitConstants::GeV;
    //Scalar deltae = delta4.w() / Acts::UnitConstants::GeV;
    // write hit index along trajectory
    //int32_t index = hit.index();
    // decoded geometry for simplicity
    /*
    m_volumeId = hit.geometryId().volume();
    m_boundaryId = hit.geometryId().boundary();
    m_layerId = hit.geometryId().layer();
    m_approachId = hit.geometryId().approach();
    m_sensitiveId = hit.geometryId().sensitive();
    */
    
    /*if (0 >= f) {
      ACTS_ERROR("f must be positive");
      return ProcessCode::ABORT;
    }*/
    double *vec = (double *) malloc( f * sizeof(double) );
    vec[0] = (double)tx;
    if (f >= 2){
        vec[1] = (double)ty;
    }
    if (f >= 3){
        vec[2] = (double)tz;
    }
    
    t.add_item(hitIndex, vec);
    
    // Fill the tree
    //m_outputTree->Fill();
    /**
    if (hitIndex == index){
      std::cout << "Hit " << index << " same index" << std::endl;
    } else {
      std::cout << "Hit " << index << " different hit index " << hitIndex << std::endl;
    }

    if ((hit.fourPosition() == hits.nth(hitIndex)->fourPosition()) && 0){
      std::cout << "Hit " << hitIndex << " same hit " << std::endl;
    } else {
      std::cout << "Hit " << hitIndex << " different hit " << std::endl;
      //int hitIndex2 = 0;
      for (auto h = hits.begin(); h != hits.end(); ++h) {
        //const auto& simHit = *h;
        const auto hitIndex2 = hits.index_of(h);
        if (hit.fourPosition() == h->fourPosition()){
          std::cout << "Hit " << hitIndex << " same hit " << hitIndex2 << std::endl;
          break;
        }
      }
    }
    //**/
    hitIndex++;
  }

  if ((unsigned int)hitIndex != hits.size()){
    //ACTS_ERROR("Not the same number of buckets than the number of hits");
    std::cout << "Not the same number of buckets than the number of hits" << std::endl;
    return ActsExamples::ProcessCode::ABORT;
  }

  unsigned int n_trees = 2 * f;
  
  t.build(n_trees);
  
  //std::cout << "Bucket Size " << bucketSize << std::endl;
  //std::cout << "Saving index ..." << std::endl;
	//t.save("precision.tree");
	//std::cout << " Done" << std::endl;



	//******************************************************

	// doing the work
	for(hitIndex=0; hitIndex < hits.size(); hitIndex++){
		//std::cout << "finding neighbours for hit " << hitIndex << " event " << eventId << std::endl;

	  SimHitContainer::sequence_type bucket;
	  std::vector<int> bucket_ids;

		// get the bucketSize closests
		t.get_nns_by_item(hitIndex, bucketSize, -1, &bucket_ids, nullptr); //search_k defaults to "n_trees * n" if not provided.

    std::sort(bucket_ids.begin(), bucket_ids.end(), std::less<int>());
    for(const auto& bucketHitidx : bucket_ids){
        bucket.push_back(*(hits.nth(bucketHitidx)));
		}

    // associate hit index with the bucket
		m_bucketsMap[hitIndex] = bucket;
    
    /**
    // write simulated hits
    CsvSimHitWriter::Config writeSimHits;
    writeSimHits.inputSimHits = "hashingBucket" + to_string(eventId) + "_" + to_string (i);
    writeSimHits.outputDir = "./csv"//outputDir;
    writeSimHits.outputStem = "hits";
    // issue: no sequencer
    sequencer.addWriter(
        std::make_shared<CsvSimHitWriter>(writeSimHits, logLevel));
    **/

	}

	//std::cout << "\nDone" << std::endl;

  return ActsExamples::ProcessCode::SUCCESS;
}

ActsExamples::ProcessCode ActsExamples::HashingAnnoy::ComputeSpacePointsBuckets(
    const AlgorithmContext& ctx, 
    const ActsExamples::SimSpacePointContainer& spacePoints,
    const unsigned int bucketSize) {
  using Scalar = Acts::ActsScalar;

  int32_t f = 2; // add it as arg; corresponds to the number of features
  int spacePointIndex = 0;
  using AnnoyMetric = Annoy::Angular;
  //using AnnoyMetric = Annoy::Euclidean;
  //-DANNOYLIB_MULTITHREADED_BUILD
  //Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy> t = 
  //Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>(f);

  Annoy::AnnoyIndex<int, double, AnnoyMetric, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy> t = 
  Annoy::AnnoyIndex<int, double, AnnoyMetric, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy>(f);


  // Get the event number
  int32_t eventId = ctx.eventNumber;
  
  // Add spacePoints parameters to Annoy
  for (const auto& spacePoint : spacePoints) {
    //int32_t geometryId = hit.geometryId().value();
    // write spacePoint position
    Scalar x = spacePoint.x() / Acts::UnitConstants::mm;
    Scalar y = spacePoint.y() / Acts::UnitConstants::mm;
    Scalar z = spacePoint.z() / Acts::UnitConstants::mm;
    //Scalar tt = hit.fourPosition().w() / Acts::UnitConstants::ns;

    // Helix transform
    Scalar r2 = x*x + y*y;
    // if (r2 > 200*200){
    // 	continue;
    // }
    //x = x / r2;
    //y = y / r2;
    /*if (0 >= f) {
      ACTS_ERROR("f must be positive");
      return ProcessCode::ABORT;
    }*/
    double *vec = (double *) malloc( f * sizeof(double) );
    vec[0] = (double)x;
    if (f >= 2){
        vec[1] = (double)y;
    }
    if (f >= 3){
        vec[2] = (double)z;
    }
    
    t.add_item(spacePointIndex, vec);
    spacePointIndex++;
  }

  if ((unsigned int)spacePointIndex != spacePoints.size()){
    //ACTS_ERROR("Not the same number of buckets than the number of hits");
    std::cout << "Not the same number of buckets than the number of spacePoints" << std::endl;
    return ActsExamples::ProcessCode::ABORT;
  }

  unsigned int n_trees = 2 * f;
  
  t.build(n_trees);
  
  //std::cout << "Bucket Size " << bucketSize << std::endl;
  //std::cout << "Saving index ..." << std::endl;
	//t.save("precision.tree");
	//std::cout << " Done" << std::endl;



	//******************************************************

  unsigned int n_buckets = 0;

	// doing the work
	for(spacePointIndex=0; spacePointIndex < spacePoints.size(); spacePointIndex++){
		//std::cout << "finding neighbours for spacePoint " << spacePointIndex << " event " << eventId << std::endl;
	  auto spacePoint = spacePoints[spacePointIndex];
    Scalar x = spacePoint.x() / Acts::UnitConstants::mm;
    Scalar y = spacePoint.y() / Acts::UnitConstants::mm;
    Scalar z = spacePoint.z() / Acts::UnitConstants::mm;
    //Scalar tt = hit.fourPosition().w() / Acts::UnitConstants::ns;

	  SimSpacePointContainer bucket;
    // Helix transform
    // Scalar r2 = x*x + y*y;
	  // std::cout << "r2:" << r2 << std::endl;
    // if (r2 > 200*200){
	  //   // std::cout << "r2 skip" << std::endl;
    // 	continue;
    // }
    // if (r2 < 33*33){
	  //   // std::cout << "r2 skip" << std::endl;
		//   // m_bucketsSPMap[spacePointIndex] = bucket;
    // 	continue;
    // }
    // if (r2 < 60*60){
	  //   // std::cout << "r2 skip" << std::endl;
		//   // m_bucketsSPMap[spacePointIndex] = bucket;
    // 	continue;
    // }
    // if (r2 > 90*90){
	  //   // std::cout << "r2 skip" << std::endl;
		//   // m_bucketsSPMap[spacePointIndex] = bucket;
    // 	continue;
    // }
    // if (r2 < 25*25){
	  //   // std::cout << "r2 skip" << std::endl;
		//   // m_bucketsSPMap[spacePointIndex] = bucket;
    // 	continue;
    // }
    // if (r2 > 40*40){
	  //   // std::cout << "r2 skip" << std::endl;
		//   // m_bucketsSPMap[spacePointIndex] = bucket;
    // 	continue;
    // }
	  // SimSpacePointContainer bucket;
	  std::vector<int> bucket_ids;

		// get the bucketSize closests spacePoints
		t.get_nns_by_item(spacePointIndex, bucketSize, -1, &bucket_ids, nullptr); //search_k defaults to "n_trees * n" if not provided.

    std::sort(bucket_ids.begin(), bucket_ids.end(), std::less<int>());
    for(const auto& bucketSpacePointIndex : bucket_ids){
        bucket.push_back(spacePoints.at(bucketSpacePointIndex));
		}

    bool found_same = false;
    for(const auto& bucketSpacePointIndex : bucket_ids){
        // std::cout << ",spacePointIndexPrev"  << spacePointIndexPrev << " spacePointIndex " << spacePointIndex << std::endl;
        if (m_bucketsSPMap.find(bucketSpacePointIndex) != m_bucketsSPMap.end()){
          SimSpacePointContainer prev_bucket = m_bucketsSPMap.find(bucketSpacePointIndex)->second;
          if (bucket == prev_bucket){
            found_same = true;
            // std::cout << "found_same" << std::endl;
            break;
          }
        }
      }
    // if (spacePointIndex > 0){
    //   if (found_same){
    //     continue;
    //   }
    //   for (unsigned int spacePointIndexPrev = spacePointIndex-1; spacePointIndexPrev >= 0; spacePointIndexPrev--){
    //     // std::cout << ",spacePointIndexPrev"  << spacePointIndexPrev << " spacePointIndex " << spacePointIndex << std::endl;
    //     if (m_bucketsSPMap.find(spacePointIndexPrev) != m_bucketsSPMap.end()){
    //       SimSpacePointContainer prev_bucket = m_bucketsSPMap.find(spacePointIndexPrev)->second;
    //       if (bucket == prev_bucket){
    //         found_same = true;
    //         std::cout << "bug found_same" << std::endl;
    //         std::cout << ",spacePointIndexPrev"  << spacePointIndexPrev << " spacePointIndex " << spacePointIndex << std::endl;
    //         break;
    //       }
    //     }
    //     if (spacePointIndexPrev == 0){
    //       break;
    //     }
    //   }
    // }
    if (found_same){
      continue;
    }

    // associate spacePoint index with the bucket
		m_bucketsSPMap[spacePointIndex] = bucket;
    n_buckets++;
	}

	//std::cout << "\nDone" << std::endl;
	std::cout << "n_buckets:" << n_buckets << std::endl;

  return ActsExamples::ProcessCode::SUCCESS;
}
