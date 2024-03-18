if (ACTS_BUILD_HASHING)
  target_sources(
    ActsCore
    PRIVATE
    Hashing/HashingAlgorithm.cpp
    Hashing/HashingAnnoy.cpp
    Hashing/HashingTraining.cpp
  )

  add_library(
  ActsExamplesHashing SHARED 
  Hashing/HashingAlgorithm.cpp
  Hashing/HashingAnnoy.cpp)

  target_include_directories(
  ActsExamplesHashing
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_link_libraries(
  ActsExamplesHashing
  PUBLIC ActsCore ActsExamplesFramework Boost::program_options)

  install(
  TARGETS ActsExamplesHashing
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  add_library(
  ActsExamplesHashingTraining SHARED 
  Hashing/HashingTraining.cpp)

  target_include_directories(
  ActsExamplesHashingTraining
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  target_link_libraries(
  ActsExamplesHashingTraining
  PUBLIC ActsCore ActsExamplesFramework Boost::program_options)

  install(
  TARGETS ActsExamplesHashingTraining
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()