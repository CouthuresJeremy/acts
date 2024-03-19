// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/Hashing/HashingAnnoy.hpp"
// #include "Acts/Seeding/Hashing/HashingTraining.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Seeding/Hashing/kissrandom.h"
#include "Acts/Seeding/Hashing/annoylib_custom.h"

#include "Acts/Definitions/Algebra.hpp"

#include <map>
#include <vector>
#include <set>

bool LayerSelection(double r2, double z){
  bool isInside = (r2 > 25*25 && r2 < 40*40) && (z > -550 && z < 550);
  // if ((r2 < 25*25 || r2 > 40*40 || (z < -550 || z > 550))
      //     && (r2 < 25*25 || r2 > 190*190 || z < -630 || z > -550)
      //     && (r2 < 25*25 || r2 > 190*190 || z < 550 || z > 630)){
  return isInside;
}

int GetBinIndex(double r2, double z, unsigned int zBins){
  using Scalar = Acts::ActsScalar;
  Scalar binSize = 1100.0/zBins;
  int binIndex = (z - (-550) + 0.5*binSize)/binSize;
  // int binIndex = (z - (-550))/binSize;
  // Scalar z = 1100.0/zBins*binIndex + -550;
  return binIndex;
}

int GetBinIndexPhi(double phi, unsigned int phiBins){
  using Scalar = Acts::ActsScalar;
  Scalar binSize = 2*M_PI/phiBins;
  int binIndex = (phi+M_PI)/binSize;
  return binIndex;
}

// template <typename AnnoyModel>
template <typename external_spacepoint_t, typename SpacePointContainer>
void Acts::HashingAnnoy<external_spacepoint_t, SpacePointContainer>::ComputeSpacePointsBuckets(
    // const AlgorithmContext& ctx, 
    // AnnoyModel* annoyModel,
    const Annoy::AnnoyIndex<unsigned int, double, Annoy::AngularEuclidean, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy>* annoyModel,
    const SpacePointContainer& spacePoints,
    const unsigned int bucketSize,
    const unsigned int zBins,
    const unsigned int phiBins) {
  using Scalar = Acts::ActsScalar;

	//******************************************************
  // std::cout << annoyModel->get_seed() << "\n";
  // std::cout << annoyModel->get_n_items() << "\n";

  if (zBins > 0){
    std::set<external_spacepoint_t> bucketsSetSPMap[zBins];
    for(unsigned int spacePointIndex=0; spacePointIndex < spacePoints.size(); spacePointIndex++){
      auto spacePoint = spacePoints[spacePointIndex];
      Scalar x = spacePoint.x() / Acts::UnitConstants::mm;
      Scalar y = spacePoint.y() / Acts::UnitConstants::mm;
      Scalar z = spacePoint.z() / Acts::UnitConstants::mm;
      //Scalar tt = hit.fourPosition().w() / Acts::UnitConstants::ns;

      // Helix transform
      Scalar r2 = x*x + y*y;
      // std::cout << "r2:" << r2 << std::endl;

      if (!LayerSelection(r2, z)){
        // std::cout << "r2 skip" << std::endl;
        // m_bucketsSPMap[spacePointIndex] = bucket;
        continue;
      }

      int binIndex = GetBinIndex(r2, z, zBins);
      if (binIndex < 0 || binIndex >= zBins){
        throw std::runtime_error("binIndex outside of bins covering");
      }
      
      // std::cout << "Here1\n";
      std::set<external_spacepoint_t> *bucketSet;
      bucketSet = &bucketsSetSPMap[binIndex];

      // SimSpacePointContainer bucket;
      std::vector<unsigned int> bucket_ids;

      /// Get the bucketSize closests spacePoints
      annoyModel->get_nns_by_item(spacePointIndex, bucketSize, -1, &bucket_ids, nullptr);

      // Scalar phi = atan2(y, x);
      // double *vec = (double *) malloc( 1 * sizeof(double) );
      // vec[0] = (double)phi;
      // annoyModel->get_nns_by_vector(vec, bucketSize, -1, &bucket_ids, nullptr);

      if (bucketSize != bucket_ids.size()){
         std::cout << "bucketSize: " << bucketSize << " returned: " << bucket_ids.size() << "\n";
      }
      for(const auto& bucketSpacePointIndex : bucket_ids){
          bucketSet->insert(spacePoints.at(bucketSpacePointIndex));
      }
      
      // m_bucketsSPMap[binIndex] = bucketSet;
    }

    unsigned int n_buckets = 0;
    for (int binIndex = 0; binIndex < zBins; binIndex++){
      if (bucketsSetSPMap[binIndex].size() > 0){
        m_bucketsSPMap[n_buckets] = bucketsSetSPMap[binIndex];
        n_buckets++;
      }
    }
  } else if (phiBins > 0) {
    std::set<external_spacepoint_t> bucketsSetSPMap[phiBins];
    for(unsigned int spacePointIndex=0; spacePointIndex < spacePoints.size(); spacePointIndex++){
      auto spacePoint = spacePoints[spacePointIndex];
      Scalar x = spacePoint.x() / Acts::UnitConstants::mm;
      Scalar y = spacePoint.y() / Acts::UnitConstants::mm;
      Scalar z = spacePoint.z() / Acts::UnitConstants::mm;
      //Scalar tt = hit.fourPosition().w() / Acts::UnitConstants::ns;

      // Helix transform
      Scalar r2 = x*x + y*y;
      // std::cout << "r2:" << r2 << std::endl;

      if (!LayerSelection(r2, z)){
        // std::cout << "r2 skip" << std::endl;
        // m_bucketsSPMap[spacePointIndex] = bucket;
        continue;
      }

      Scalar phi = atan2(y, x);

      int binIndex = GetBinIndexPhi(phi, phiBins);
      if (binIndex < 0 || binIndex >= phiBins){
        throw std::runtime_error("binIndex outside of bins covering");
      }
      
      // std::cout << "Here1\n";
      std::set<external_spacepoint_t> *bucketSet;
      bucketSet = &bucketsSetSPMap[binIndex];

      // SimSpacePointContainer bucket;
      std::vector<unsigned int> bucket_ids;

      /// Get the bucketSize closests spacePoints
      annoyModel->get_nns_by_item(spacePointIndex, bucketSize, -1, &bucket_ids, nullptr);

      // Scalar phi = atan2(y, x);
      // double *vec = (double *) malloc( 1 * sizeof(double) );
      // vec[0] = (double)phi;
      // annoyModel->get_nns_by_vector(vec, bucketSize, -1, &bucket_ids, nullptr);

      if (bucketSize != bucket_ids.size()){
         std::cout << "bucketSize: " << bucketSize << " returned: " << bucket_ids.size() << "\n";
      }
      for(const auto& bucketSpacePointIndex : bucket_ids){
          bucketSet->insert(spacePoints.at(bucketSpacePointIndex));
      }
      
      // m_bucketsSPMap[binIndex] = bucketSet;
    }

    unsigned int n_buckets = 0;
    for (int binIndex = 0; binIndex < phiBins; binIndex++){
      if (bucketsSetSPMap[binIndex].size() > 0){
        m_bucketsSPMap[n_buckets] = bucketsSetSPMap[binIndex];
        n_buckets++;
      }
    }
  }
  else {
  }

  // return Acts::ProcessCode::SUCCESS;
}