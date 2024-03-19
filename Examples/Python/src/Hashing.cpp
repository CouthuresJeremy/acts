// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/Python/Utilities.hpp"
#include "Acts/Seeding/Hashing/HashingAlgorithm.hpp"

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// using namespace ActsExamples;
using namespace Acts;

namespace Acts::Python {

void addHashing(Context& ctx) {
  auto mex = ctx.get("examples");
  // auto [m, mex] = ctx.get("main", "examples");

  // ACTS_PYTHON_DECLARE_ALGORITHM(
  //     ActsExamples::HashingAlgorithm, mex, "HashingAlgorithm", inputSpacePoints,
  //     bucketSize, zBins, phiBins, outputBuckets);

  // {
  //   using Alg = ActsExamples::HashingAlgorithm;
  //   using Config = Alg::Config;

  //   auto alg =
  //       py::class_<ActsExamples::HashingAlgorithm, ActsExamples::BareAlgorithm,
  //                  std::shared_ptr<ActsExamples::HashingAlgorithm>>(
  //           mex, "HashingAlgorithm")
  //           .def(py::init<const Config&, Acts::Logging::Level>(),
  //                py::arg("config"), py::arg("level"))
  //           .def_property_readonly("config",
  //                                  &ActsExamples::HashingAlgorithm::config);

  //   auto c = py::class_<Config>(alg, "Config").def(py::init<>());

  //   ACTS_PYTHON_STRUCT_BEGIN(c, Config);
  //   ACTS_PYTHON_MEMBER(inputSpacePoints);
  //   ACTS_PYTHON_MEMBER(bucketSize);
  //   ACTS_PYTHON_MEMBER(zBins);
  //   ACTS_PYTHON_MEMBER(phiBins);
  //   ACTS_PYTHON_MEMBER(outputBuckets);
  //   ACTS_PYTHON_STRUCT_END();
  // }

  // // auto [m, mex, onnx] = ctx.get("main", "examples", "onnx");
  // // auto mlpack = mex.def_submodule("_mlpack");

  // ACTS_PYTHON_DECLARE_ALGORITHM(
  //     ActsExamples::AmbiguityResolutionMLDBScanAlgorithm, mlpack,
  //     "AmbiguityResolutionMLDBScanAlgorithm", inputTracks, inputDuplicateNN,
  //     outputTracks, nMeasurementsMin, epsilonDBScan, minPointsDBScan);

}

}  // namespace Acts::Python
