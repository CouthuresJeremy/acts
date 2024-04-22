// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Io/Csv/CsvBucketWriter.hpp"

#include "Acts/Definitions/Units.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Utilities/Paths.hpp"

#include <cstddef>
#include <ios>
#include <optional>
#include <stdexcept>

#include <dfe/dfe_io_dsv.hpp>

#include "CsvOutputData.hpp"

ActsExamples::CsvBucketWriter::CsvBucketWriter(
    const ActsExamples::CsvBucketWriter::Config& config,
    Acts::Logging::Level level)
    : WriterT(config.inputBuckets, "CsvBucketWriter", level), m_cfg(config) {}

ActsExamples::CsvBucketWriter::~CsvBucketWriter() = default;

ActsExamples::ProcessCode ActsExamples::CsvBucketWriter::finalize() {
  // Write the tree
  return ProcessCode::SUCCESS;
}

ActsExamples::ProcessCode ActsExamples::CsvBucketWriter::writeT(
    const AlgorithmContext& ctx,
    const std::vector<SimSpacePointContainer>& buckets) {
  // Open per-event file for all components
  std::string pathBucket =
      perEventFilepath(m_cfg.outputDir, "buckets.csv", ctx.eventNumber);

  dfe::NamedTupleCsvWriter<BucketData> writerBucket(pathBucket,
                                                    m_cfg.outputPrecision);

  BucketData bucketData{};

  int bucketIdx = 0;
  for (const auto& bucket : buckets) {
    if (bucket.empty()) {
      continue;
    }
    // Split the bucket into lines of 20 space points to manage variable sizes
    auto numLines =
        static_cast<int>((bucket.size() / 20) +
                         static_cast<std::size_t>(bucket.size() % 20 != 0));
    for (int nLines = 0; nLines < numLines; nLines++) {
      bucketData.bucketIdx = bucketIdx;
      bucketData.bucketSize = bucket.size();
      int maxSPIdx =
          std::min(20, static_cast<int>(bucket.size() - nLines * 20));
      for (int SPIdx = 0; SPIdx < maxSPIdx; SPIdx++) {
        bucketData.measurement_id[SPIdx] = (bucket[nLines * 20 + SPIdx])
                                               .sourceLinks()[0]
                                               .get<IndexSourceLink>()
                                               .index();
      }
      writerBucket.append(bucketData);
    }
    bucketIdx++;
  }

  return ActsExamples::ProcessCode::SUCCESS;
}
