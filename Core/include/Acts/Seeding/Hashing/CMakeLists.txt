# add_library(
#   ActsExamplesHashing SHARED 
#   src/HashingAlgorithm.cpp
#   src/HashingAnnoy.cpp)

# target_include_directories(
#   ActsExamplesHashing
#   PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# target_link_libraries(
#   ActsExamplesHashing
#   PUBLIC ActsCore ActsExamplesFramework Boost::program_options)

# install(
#   TARGETS ActsExamplesHashing
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# add_library(
#   ActsExamplesHashingTraining SHARED 
#   src/HashingTraining.cpp)

# target_include_directories(
#   ActsExamplesHashingTraining
#   PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# target_link_libraries(
#   ActsExamplesHashingTraining
#   PUBLIC ActsCore ActsExamplesFramework Boost::program_options)

# install(
#   TARGETS ActsExamplesHashingTraining
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

