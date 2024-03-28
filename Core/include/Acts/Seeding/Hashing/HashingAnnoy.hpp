#include "Acts/Utilities/Logger.hpp"
// #include "ActsExamples/Framework/AlgorithmContext.hpp"
// #include "ActsExamples/Framework/ProcessCode.hpp"
// #include "ActsExamples/EventData/SimSpacePoint.hpp"
// #include "Acts/Seeding/InternalSpacePoint.hpp"

#include "Acts/Seeding/Hashing/kissrandom.h"
#include "Acts/Seeding/Hashing/annoylib_custom.h"

#include <map>
#include <set>

namespace Acts {

template <typename external_spacepoint_t, typename SpacePointContainer>
class HashingAnnoy {
  public:
    void ComputeSpacePointsBuckets(
      // const AlgorithmContext& ctx, 
      const Annoy::AnnoyIndex<unsigned int, double, Annoy::AngularEuclidean, Annoy::Kiss32Random, 
                    Annoy::AnnoyIndexSingleThreadedBuildPolicy>* annoyModel,
      const SpacePointContainer& spacePoints,
      const unsigned int bucketSize,
      const unsigned int zBins,
      const unsigned int phiBins
      );
    std::map<int, std::set<external_spacepoint_t>> m_bucketsSPMap;
};
}
#include "Acts/Seeding/Hashing/HashingAnnoy.ipp"