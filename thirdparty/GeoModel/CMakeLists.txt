# This file is part of the Acts project.
#
# Copyright (C) 2024 CERN for the benefit of the Acts project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# CMake include(s).
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building GeoModel as part of the ACTS project" )

set( GEOMODEL_VERSION "${_acts_geomodel_version}")

set(GEOMODEL_SETUP_JSON OFF CACHE BOOL "Skip setting up json completely" )

set(GEOMODEL_BUILD_GEOMODELG4 "${ACTS_BUILD_PLUGIN_GEANT4}" CACHE BOOL "Build the Geant4 plugin" )

# Declare where to get geomodel from.
FetchContent_Declare(geomodel ${ACTS_GEOMODEL_SOURCE}
    PATCH_COMMAND git am ${CMAKE_CURRENT_SOURCE_DIR}/0001-Add-option-to-skip-setting-up-json-completely.patch)

# Now set up its build.
FetchContent_MakeAvailable( geomodel )
