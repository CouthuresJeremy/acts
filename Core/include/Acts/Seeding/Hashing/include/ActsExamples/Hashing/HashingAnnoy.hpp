#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"

#include "ActsExamples/Hashing/kissrandom.h"
#include "ActsExamples/Hashing/annoylib_custom.h"

#include <map>
#include <set>

namespace ActsExamples {
class HashingAnnoy {
  public:
    ActsExamples::ProcessCode ComputeSpacePointsBuckets(
      const AlgorithmContext& ctx, 
      const Annoy::AnnoyIndex<unsigned int, double, Annoy::AngularEuclidean, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy>* annoyModel,
      const ActsExamples::SimSpacePointContainer& spacePoints,
      const unsigned int bucketSize,
      const unsigned int zBins,
      const unsigned int phiBins
      );
    std::map<int, std::set<ActsExamples::SimSpacePoint>> m_bucketsSPMap;
};
}