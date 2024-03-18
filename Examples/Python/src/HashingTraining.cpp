// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/Python/Utilities.hpp"
#include "Acts/Seeding/Hashing/include/HashingTraining.hpp"

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace ActsExamples;
using namespace Acts;

namespace Acts::Python {

void addHashingTraining(Context& ctx) {
  auto mex = ctx.get("examples");


  ACTS_PYTHON_DECLARE_ALGORITHM(
      ActsExamples::HashingTrainingAlgorithm, mex, "HashingTrainingAlgorithm", inputSpacePoints,
      AnnoySeed, f);

  // {
  //   using Alg = ActsExamples::HashingTrainingAlgorithm;
  //   using Config = Alg::Config;

  //   auto alg =
  //       py::class_<ActsExamples::HashingTrainingAlgorithm, ActsExamples::BareAlgorithm,
  //                  std::shared_ptr<ActsExamples::HashingTrainingAlgorithm>>(
  //           mex, "HashingTrainingAlgorithm")
  //           .def(py::init<const Config&, Acts::Logging::Level>(),
  //                py::arg("config"), py::arg("level"))
  //           .def_property_readonly("config",
  //                                  &ActsExamples::HashingTrainingAlgorithm::config);

  //   auto c = py::class_<Config>(alg, "Config").def(py::init<>());

  //   ACTS_PYTHON_STRUCT_BEGIN(c, Config);
  //   ACTS_PYTHON_MEMBER(inputSpacePoints);
  //   ACTS_PYTHON_MEMBER(AnnoySeed);
  //   ACTS_PYTHON_MEMBER(f);
  //   ACTS_PYTHON_STRUCT_END();
  // }

}

}  // namespace Acts::Python
