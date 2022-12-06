// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/Python/Utilities.hpp"
#include "ActsExamples/Hashing/MergeSeedsAlgorithm.hpp"

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace ActsExamples;
using namespace Acts;

namespace Acts::Python {

void mergeSeeds(Context& ctx) {
  auto mex = ctx.get("examples");
 
  {
    using Alg = ActsExamples::MergeSeedsAlgorithm;
    using Config = Alg::Config;

    auto alg =
        py::class_<ActsExamples::MergeSeedsAlgorithm, ActsExamples::BareAlgorithm,
                   std::shared_ptr<ActsExamples::MergeSeedsAlgorithm>>(
            mex, "MergeSeedsAlgorithm")
            .def(py::init<const Config&, Acts::Logging::Level>(),
                 py::arg("config"), py::arg("level"))
            .def_property_readonly("config",
                                   &ActsExamples::MergeSeedsAlgorithm::config);

    auto c = py::class_<Config>(alg, "Config").def(py::init<>());

    ACTS_PYTHON_STRUCT_BEGIN(c, Config);
    ACTS_PYTHON_MEMBER(inputSeeds);
    ACTS_PYTHON_MEMBER(inputProtoTracks);
    ACTS_PYTHON_MEMBER(outputSeeds);
    ACTS_PYTHON_MEMBER(outputProtoTracks);
    ACTS_PYTHON_STRUCT_END();
  }

}

}  // namespace Acts::Python
