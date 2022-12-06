#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/EventData/SimHit.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"

#include <map>

namespace ActsExamples {
class HashingAnnoy {
  public:
    //HitsPrinter(
    //const ActsExamples::HitsPrinter::Config& cfg, Acts::Logging::Level level);
    ActsExamples::ProcessCode ComputeHitBuckets(
    const AlgorithmContext& ctx, 
    const ActsExamples::SimHitContainer& hits,
    const unsigned int bucketSize);

    ActsExamples::ProcessCode ComputeSpacePointsBuckets(
    const AlgorithmContext& ctx, 
    const ActsExamples::SimSpacePointContainer& spacePoints,
    const unsigned int bucketSize);
    //std::map<int, std::vector<int>> m_bucketsMap;
    std::map<int, SimHitContainer::sequence_type> m_bucketsMap;
    std::map<int, SimSpacePointContainer> m_bucketsSPMap;
};
}
