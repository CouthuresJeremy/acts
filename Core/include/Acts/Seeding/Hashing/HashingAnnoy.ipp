// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

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

namespace Acts {
int GetBinIndex(double, double z, unsigned int zBins){
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
void HashingAnnoy<external_spacepoint_t, SpacePointContainer>::ComputeSpacePointsBuckets(
    // const AlgorithmContext& ctx, 
    // AnnoyModel* annoyModel,
    const Annoy::AnnoyIndex<unsigned int, double, Annoy::AngularEuclidean, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy>* annoyModel,
    const SpacePointContainer& spacePoints,
    const unsigned int bucketSize,
    const unsigned int zBins,
    const unsigned int phiBins) {
  using Scalar = Acts::ActsScalar;

  static thread_local std::vector<std::set<external_spacepoint_t>> bucketsSetSPMap;
  bucketsSetSPMap.clear();
  
  unsigned int nBins = 0;
  if (zBins > 0){
    nBins = zBins;
  } else if (phiBins > 0){
    nBins = phiBins;
  } else {
    throw std::runtime_error("No bins defined");
  }
  bucketsSetSPMap.reserve(nBins);

  // Create the set of spacePoints for each bucket
  for (unsigned int binIndex = 0; binIndex < nBins; binIndex++){
    static thread_local std::set<external_spacepoint_t> bucket;
    bucket.clear();
    bucketsSetSPMap.push_back(bucket);
  }

  if (zBins > 0){
    // Loop over spacePoints
    for(unsigned int spacePointIndex=0; spacePointIndex < spacePoints.size(); spacePointIndex++){
      external_spacepoint_t spacePoint = spacePoints[spacePointIndex];
      Scalar x = spacePoint->x() / Acts::UnitConstants::mm;
      Scalar y = spacePoint->y() / Acts::UnitConstants::mm;
      Scalar z = spacePoint->z() / Acts::UnitConstants::mm;
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
      if (binIndex < 0 || (unsigned int)binIndex >= nBins){
        throw std::runtime_error("binIndex outside of bins covering");
      }

      static thread_local std::vector<unsigned int> bucket_ids;
      bucket_ids.clear();

      /// Get the bucketSize closest spacePoints
      annoyModel->get_nns_by_item(spacePointIndex, bucketSize, -1, &bucket_ids, nullptr);

      // Scalar phi = atan2(y, x);
      // double *vec = (double *) malloc( 1 * sizeof(double) );
      // vec[0] = (double)phi;
      // annoyModel->get_nns_by_vector(vec, bucketSize, -1, &bucket_ids, nullptr);

      if (bucketSize != bucket_ids.size()){
         std::cout << "bucketSize: " << bucketSize << " returned: " << bucket_ids.size() << "\n";
      }
      for(const unsigned int& bucketSpacePointIndex : bucket_ids){
          bucketsSetSPMap[(unsigned int)binIndex].insert(spacePoints.at(bucketSpacePointIndex));
      }
    }
  } else if (phiBins > 0) {
    // Loop over spacePoints
    for(unsigned int spacePointIndex=0; spacePointIndex < spacePoints.size(); spacePointIndex++){
      external_spacepoint_t spacePoint = spacePoints[spacePointIndex];
      Scalar x = spacePoint->x() / Acts::UnitConstants::mm;
      Scalar y = spacePoint->y() / Acts::UnitConstants::mm;
      Scalar z = spacePoint->z() / Acts::UnitConstants::mm;
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
      if (binIndex < 0 || (unsigned int)binIndex >= nBins){
        throw std::runtime_error("binIndex outside of bins covering");
      }

      std::vector<unsigned int> bucket_ids;

      /// Get the bucketSize closest spacePoints
      annoyModel->get_nns_by_item(spacePointIndex, bucketSize, -1, &bucket_ids, nullptr);

      // Scalar phi = atan2(y, x);
      // double *vec = (double *) malloc( 1 * sizeof(double) );
      // vec[0] = (double)phi;
      // annoyModel->get_nns_by_vector(vec, bucketSize, -1, &bucket_ids, nullptr);

      if (bucketSize != bucket_ids.size()){
         std::cout << "bucketSize: " << bucketSize << " returned: " << bucket_ids.size() << "\n";
      }
      for(const unsigned int& bucketSpacePointIndex : bucket_ids){
          bucketsSetSPMap[(unsigned int)binIndex].insert(spacePoints.at(bucketSpacePointIndex));
      }
    }
  }
  else {
    throw std::runtime_error("No bins defined");
  }

  unsigned int n_buckets = 0;
  for (unsigned int binIndex = 0; binIndex < nBins; binIndex++){
    if (bucketsSetSPMap[binIndex].size() > 0){
      m_bucketsSPMap[n_buckets] = bucketsSetSPMap[binIndex];
      n_buckets++;
    }
  }
}

}  // namespace Acts
