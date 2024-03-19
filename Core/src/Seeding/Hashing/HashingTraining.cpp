// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Seeding/Hashing/HashingTraining.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Definitions/Algebra.hpp"
// #include "ActsExamples/Framework/WhiteBoard.hpp"

template <typename SpacePointContainer>
Acts::HashingTrainingAlgorithm<SpacePointContainer>::HashingTrainingAlgorithm(
    const Acts::HashingTrainingAlgorithmConfig& cfg, 
    Acts::Logging::Level level)
    : m_cfg(cfg) {
  if (m_cfg.f <= 0) {
    throw std::invalid_argument("Invalid f, f must be positive");
  }
  // if (m_cfg.inputSpacePoints.empty()) {
  //   throw std::invalid_argument("Missing space points input collections");
  // }
  if (m_cfg.AnnoySeed <= 0) {
    throw std::invalid_argument("Invalid Annoy random seed, Annoy random seed must be positive");
  }
  // if (m_cfg.inputSpacePoints.empty()) {
  //   throw std::invalid_argument("Missing space point input collections");
  // }
  // m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);

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
  // if (m_cfg.outputSeeds.empty()) {
  //   throw std::invalid_argument("Missing seeds output collection");
  // }

  // m_outputSeeds.initialize(m_cfg.outputSeeds);
  // m_outputAnnoyModel.initialize("OutputAnnoyModel");
}

using AnnoyMetric = Annoy::AngularEuclidean;
//-DANNOYLIB_MULTITHREADED_BUILD
//Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy> annoyModel = 
//Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>(f);

using AnnoyModel = Annoy::AnnoyIndex<unsigned int, double, AnnoyMetric, Annoy::Kiss32Random, 
                  Annoy::AnnoyIndexSingleThreadedBuildPolicy>;

template <typename SpacePointContainer>
AnnoyModel Acts::HashingTrainingAlgorithm<SpacePointContainer>::execute(
    // const Acts::AlgorithmContext& ctx,
    SpacePointContainer spacePoints) const {

  // ACTS_DEBUG("event " << ctx.eventNumber);

  // const auto& spacePoints =
  //     ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);
  // const auto& spacePoints = m_inputSpacePoints(ctx);

  const unsigned int AnnoySeed = m_cfg.AnnoySeed;
  const int32_t f = m_cfg.f;

  AnnoyModel annoyModel = AnnoyModel(f);
  
  using Scalar = Acts::ActsScalar;

  annoyModel.set_seed(AnnoySeed);
  // ACTS_DEBUG("f:" << f);
  // ACTS_DEBUG("annoyModel seed:" << annoyModel.get_seed());
  
  unsigned int spacePointIndex = 0;
  // Add spacePoints parameters to Annoy
  for (const auto& spacePoint : spacePoints) {
    Scalar x = spacePoint.x() / Acts::UnitConstants::mm;
    Scalar y = spacePoint.y() / Acts::UnitConstants::mm;

    // Helix transform
    Scalar phi = atan2(y, x);

    double *vec = (double *) malloc( f * sizeof(double) );
    vec[0] = (double)phi;
    // vec[0] = (double)argument;
    // vec[0] = (double)eta;
    if (f >= 2){
        Scalar z = spacePoint.z() / Acts::UnitConstants::mm;
        Scalar r2 = x*x + y*y;
        Scalar rho = sqrt(r2 + z*z);
        Scalar theta = acos(z/rho);
        Scalar eta = -log(tan(0.5*theta));
        vec[1] = (double)eta;
    }
    
    // std::cout << "Theta " << theta << " Eta " << eta << " z " << z << std::endl;
    
    annoyModel.add_item(spacePointIndex, vec);
    spacePointIndex++;
  }

  unsigned int n_trees = 2 * f;
  
  annoyModel.build(n_trees);

  //std::cout << "Saving index ..." << std::endl;
	//annoyModel->save("precision.tree");
	//std::cout << " Done" << std::endl;

  // m_outputAnnoyModel(ctx, std::move(annoyModel));

  return annoyModel;
}
