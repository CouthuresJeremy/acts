// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Hashing/MergeSeedsAlgorithm.hpp"

#include "Acts/Digitization/PlanarModuleCluster.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Utilities/Range.hpp"
#include "Acts/Seeding/Seed.hpp"

#include <vector>
#include <set>

bool CompareSeeds (Acts::Seed<ActsExamples::SimSpacePoint> mainSeed, Acts::Seed<ActsExamples::SimSpacePoint> secondSeed);
bool CompareSP(const ActsExamples::SimSpacePoint& lhs, const ActsExamples::SimSpacePoint& rhs);

ActsExamples::MergeSeedsAlgorithm::MergeSeedsAlgorithm(
    const ActsExamples::MergeSeedsAlgorithm::Config& cfg, Acts::Logging::Level level)
    : BareAlgorithm("MergeSeedsAlgorithm", level), m_cfg(cfg) {
  if (m_cfg.inputSeeds.empty()) {
    throw std::invalid_argument("Missing seed input collections");
  }
  for (const auto& i : m_cfg.inputSeeds) {
    if (i.empty()) {
      throw std::invalid_argument("Invalid seed input collection");
    }
  }
  if (m_cfg.inputProtoTracks.empty()) {
    throw std::invalid_argument("Missing proto track input collections");
  }
  for (const auto& i : m_cfg.inputProtoTracks) {
    if (i.empty()) {
      throw std::invalid_argument("Invalid proto track input collection");
    }
  }
  if (m_cfg.outputProtoTracks.empty()) {
    throw std::invalid_argument("Missing proto tracks output collection");
  }
  if (m_cfg.outputSeeds.empty()) {
    throw std::invalid_argument("Missing seeds output collection");
  }
}

ActsExamples::ProcessCode ActsExamples::MergeSeedsAlgorithm::execute(
    const ActsExamples::AlgorithmContext& ctx) const {

  ACTS_INFO("event " << ctx.eventNumber);

  static thread_local SimSeedContainer mergedSeeds;
  mergedSeeds.clear();
  /*
  for (const auto& inputSeedsName : m_cfg.inputSeeds){
    const SimSeedContainer& bucketSeeds = ctx.eventStore.get<SimSeedContainer>(inputSeedsName);
    for (const auto& seed : bucketSeeds){
      bool found = false;
      for (const auto& secondSeed : mergedSeeds){
        found = CompareSeeds(secondSeed, seed);
        if (found){
            break;
        }
      }
      if (found){
        continue;
      }
      mergedSeeds.push_back(seed);
    }
  }
  */

  std::set<std::pair<unsigned int, std::pair<unsigned int, unsigned int> >> seedSet;

  bool first = true;
  ///*
  for (const auto& inputSeedsName : m_cfg.inputSeeds){
    const SimSeedContainer& bucketSeeds = ctx.eventStore.get<SimSeedContainer>(inputSeedsName);
    for (const auto& seed : bucketSeeds){
      //const auto slink =
      //      static_cast<const IndexSourceLink&>(*(sp.sourceLinks()[0]));

      //int measurement_id_1 = (static_cast<const IndexSourceLink&>(*((*seed.sp()[0]).sourceLinks()[0]))).index();
      unsigned int sp_index_1 = (static_cast<const IndexSourceLink&>(*((*seed.sp()[0]).sourceLinks()[0]))).index();
      unsigned int sp_index_2 = (static_cast<const IndexSourceLink&>(*((*seed.sp()[1]).sourceLinks()[0]))).index();
      unsigned int sp_index_3 = (static_cast<const IndexSourceLink&>(*((*seed.sp()[2]).sourceLinks()[0]))).index();
      if (first){
        std::cout << sp_index_1 << "," << sp_index_2 << "," << sp_index_3 << "\n";
        first = false;
      }
      // tri des sp
      // même seeds -> forcément rangées dans le même ordre
      // vérifier qu'il ne peut pas y avoir de seeds avec les space points dans un ordre différent

      auto count1 = seedSet.count(std::make_pair(
        sp_index_1,
        std::make_pair(sp_index_2,
        sp_index_3))
      );

      auto count2 = seedSet.count(std::make_pair(
        sp_index_1,
        std::make_pair(sp_index_3,
        sp_index_2))
      );

      auto count3 = seedSet.count(std::make_pair(
        sp_index_2,
        std::make_pair(sp_index_1,
        sp_index_3))
      );

      auto count4 = seedSet.count(std::make_pair(
        sp_index_2,
        std::make_pair(sp_index_3,
        sp_index_1))
      );

      auto count5 = seedSet.count(std::make_pair(
        sp_index_3,
        std::make_pair(sp_index_1,
        sp_index_2))
      );

      auto count6 = seedSet.count(std::make_pair(
        sp_index_3,
        std::make_pair(sp_index_2,
        sp_index_1))
      );

      auto ret = seedSet.emplace(
        sp_index_1,
        std::make_pair(sp_index_2,
        sp_index_3)
      );

      auto ret2 = seedSet.emplace(
        sp_index_1,
        std::make_pair(sp_index_3,
        sp_index_2)
      );

      auto ret3 = seedSet.emplace(
        sp_index_2,
        std::make_pair(sp_index_1,
        sp_index_3)
      );

      auto ret4 = seedSet.emplace(
        sp_index_2,
        std::make_pair(sp_index_3,
        sp_index_1)
      );

      auto ret5 = seedSet.emplace(
        sp_index_3,
        std::make_pair(sp_index_1,
        sp_index_2)
      );

      auto ret6 = seedSet.emplace(
        sp_index_3,
        std::make_pair(sp_index_2,
        sp_index_1)
      );

      /*
      bool found = false;
      for (const auto& secondSeed : mergedSeeds){
        found = CompareSeeds(secondSeed, seed);
        if (found){
            break;
        }
      }
      */

      //if not found
      if (ret.second && ret2.second && ret3.second && ret4.second && ret5.second && ret6.second){
        /*
        if (found){
          std::cout << "found\n";
          std::cout << count1 << "," << count2 << "," << count3 << "," << count4 << "," << count5 << "," << count6 << "\n";
        }
        */
        //if (first & sp_index_1==4261 & 14203,22184)
        mergedSeeds.push_back(seed);
      }
    }
  }
  //*/
  /*
  Acts::Vector3 globalPos1(mergedSeeds[0].sp()[0]->x(),
                           mergedSeeds[0].sp()[0]->y(),
                           mergedSeeds[0].sp()[0]->z());
  ActsExamples::SimSpacePoint sp1(globalPos1,
                                  mergedSeeds[0].sp()[0]->varianceR(),
                                  mergedSeeds[0].sp()[0]->varianceZ(),
                                  mergedSeeds[0].sp()[0]->measurementIndex());
  Acts::Vector3 globalPos2(mergedSeeds[0].sp()[1]->x(),
                           mergedSeeds[0].sp()[1]->y(),
                           mergedSeeds[0].sp()[1]->z());
  ActsExamples::SimSpacePoint sp2(globalPos2,
                                  mergedSeeds[0].sp()[1]->varianceR(),
                                  mergedSeeds[0].sp()[1]->varianceZ(),
                                  mergedSeeds[0].sp()[1]->measurementIndex());
  Acts::Vector3 globalPos3(mergedSeeds[0].sp()[2]->x(),
                           mergedSeeds[0].sp()[2]->y(),
                           mergedSeeds[0].sp()[2]->z());
  ActsExamples::SimSpacePoint sp3(globalPos3,
                                  mergedSeeds[0].sp()[2]->varianceR(),
                                  mergedSeeds[0].sp()[2]->varianceZ(),
                                  mergedSeeds[0].sp()[2]->measurementIndex());

  float z = mergedSeeds[0].z();
  Acts::Seed<ActsExamples::SimSpacePoint> customSeed(sp1, sp2, sp3, z);
  ACTS_INFO("compare sp with custom " << CompareSP(sp1, *mergedSeeds[0].sp()[0]));
  ACTS_INFO("compare seed with custom " << CompareSeeds(customSeed, mergedSeeds[0]));

  float nDuplicatedSeeds = 0.;
  unsigned int nExcessSeeds = 0.;
  for (const auto& mainSeed : mergedSeeds){
    unsigned int nSame = 0; 
    for (const auto& secondSeed : mergedSeeds){

        if (CompareSeeds(secondSeed, mainSeed)){
            nSame++;
        }
    }
    //ACTS_INFO("nSame " << nSame);
    if (nSame > 1){
        nDuplicatedSeeds += 1./nSame;
        nExcessSeeds++;
    }
  }
  nExcessSeeds -= nDuplicatedSeeds;
  ACTS_INFO("nDuplicatedSeeds " << nDuplicatedSeeds);
  ACTS_INFO("nExcessSeeds " << nExcessSeeds);
  */

  size_t nSeeds = mergedSeeds.size();
  ACTS_INFO("event " << ctx.eventNumber << " nSeeds " << nSeeds);
  static thread_local ProtoTrackContainer mergedProtoTracks;
  mergedProtoTracks.clear();

  mergedProtoTracks.reserve(nSeeds);
  for (const auto& inputProtoTrackName : m_cfg.inputProtoTracks){
    const ProtoTrackContainer& bucketProtoTracks = ctx.eventStore.get<ProtoTrackContainer>(inputProtoTrackName);
    for (const auto& protoTrack : bucketProtoTracks) {
      mergedProtoTracks.emplace_back(protoTrack);
    }
  }

  ctx.eventStore.add(m_cfg.outputSeeds, std::move(mergedSeeds));
  ctx.eventStore.add(m_cfg.outputProtoTracks, std::move(mergedProtoTracks));

  return ProcessCode::SUCCESS;
}

bool CompareSeeds (Acts::Seed<ActsExamples::SimSpacePoint> mainSeed, Acts::Seed<ActsExamples::SimSpacePoint> secondSeed) {
if (!( *mainSeed.sp()[0] == *secondSeed.sp()[0] || *mainSeed.sp()[0] == *secondSeed.sp()[1] || *mainSeed.sp()[0] == *secondSeed.sp()[2])) return false;
if (!( *mainSeed.sp()[1] == *secondSeed.sp()[0] || *mainSeed.sp()[1] == *secondSeed.sp()[1] || *mainSeed.sp()[1] == *secondSeed.sp()[2])) return false;
if (!( *mainSeed.sp()[2] == *secondSeed.sp()[0] || *mainSeed.sp()[2] == *secondSeed.sp()[1] || *mainSeed.sp()[2] == *secondSeed.sp()[2])) return false;

return true;

}

bool CompareSP(const ActsExamples::SimSpacePoint& lhs, const ActsExamples::SimSpacePoint& rhs) {
  // TODO would it be sufficient to check just the index under the assumption
  //   that the same measurement index always produces the same space point?
  // no need to check r since it is fully defined by x/y
  return ((lhs.sourceLinks() == rhs.sourceLinks()) and (lhs.x() == rhs.x()) and
          (lhs.y() == rhs.y()) and (lhs.z() == rhs.z()) and
          (lhs.varianceR() == rhs.varianceR()) and
          (lhs.varianceZ() == rhs.varianceZ()));
}
