// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Io/Csv/CsvSimSeedWriter.hpp"

#include "Acts/Definitions/Units.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/Utilities/Paths.hpp"

#include <ios>
#include <optional>
#include <stdexcept>

#include <dfe/dfe_io_dsv.hpp>

#include "CsvOutputData.hpp"

ActsExamples::CsvSimSeedWriter::CsvSimSeedWriter(
    const ActsExamples::CsvSimSeedWriter::Config& config,
    Acts::Logging::Level level)
    : WriterT(config.inputSeeds, "CsvSimSeedWriter", level),
      m_cfg(config) {}

ActsExamples::CsvSimSeedWriter::~CsvSimSeedWriter() {}

ActsExamples::ProcessCode ActsExamples::CsvSimSeedWriter::endRun() {
  // Write the tree
  return ProcessCode::SUCCESS;
}

ActsExamples::ProcessCode ActsExamples::CsvSimSeedWriter::writeT(
    const AlgorithmContext& ctx, const SimSeedContainer& simSeeds) {
  // Open per-event file for all components
  std::string pathSeed =
      perEventFilepath(m_cfg.outputDir, "seeds.csv", ctx.eventNumber);

  dfe::NamedTupleCsvWriter<SimSeedData> writerSimSeed(pathSeed,
                                                    m_cfg.outputPrecision);

  SimSeedData simSeedData;
  for (const auto& seed : simSeeds) {
    simSeedData.measurement_id_bottom = (static_cast<const IndexSourceLink&>(*(*seed.sp()[0]).sourceLinks()[0])).index();
    simSeedData.measurement_id_middle = (static_cast<const IndexSourceLink&>(*(*seed.sp()[1]).sourceLinks()[0])).index();
    simSeedData.measurement_id_top = (static_cast<const IndexSourceLink&>(*(*seed.sp()[2]).sourceLinks()[0])).index();
    writerSimSeed.append(simSeedData);
  }
  return ActsExamples::ProcessCode::SUCCESS;
}
