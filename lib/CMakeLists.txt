add_subdirectory(lib/kokkos)
add_subdirectory(lib/kokkos-kernels)

include(ExternalProject)
include(FetchContent)

FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG "release-1.11.0"
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest
  UPDATE_COMMAND ""
)
FetchContent_MakeAvailable(googletest)

# grabs a copy of nlohmann/json from a lighter repository
# which tracks the original, minus the large test data set.
# This is recommended by the readme of nlohmann/json
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/json_src
    UPDATE_COMMAND ""
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
	Spglib
	GIT_REPOSITORY https://github.com/spglib/spglib
	GIT_TAG develop
	UPDATE_COMMAND ""
)
set(SPGLIB_USE_OMP ON)
FetchContent_MakeAvailable(Spglib)

ExternalProject_Add(pugixml_dep
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/pugixml_build
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pugixml_src
    GIT_REPOSITORY https://github.com/zeux/pugixml.git
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
	-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
)

ExternalProject_Add(eigen_dep
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/eigen_src
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)
if(HDF5_AVAIL)
  if(HDF5_SERIAL)
    set(PARALLEL_HDF5 OFF)
  else()
    set(PARALLEL_HDF5 ${MPI_AVAIL})
  endif()
  ExternalProject_Add(highfive_dep
      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/highfive_src
      GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
      GIT_TAG v3.0.0-beta2
      UPDATE_COMMAND ""
      CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/highfive_src
          -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/highfive_src
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
          -DHIGHFIVE_BUILD_DOCS=OFF
          -DHIGHFIVE_EXAMPLES=OFF
          -DHIGHFIVE_UNIT_TESTS=OFF
          -DHIGHFIVE_TEST_BOOST=OFF
          -DHIGHFIVE_TEST_EIGEN=OFF
          -DHDF5_IS_PARALLEL=${PARALLEL_HDF5}
          -DHIGHFIVE_TEST_OPENCV=OFF
          -DHIGHFIVE_TEST_XTENSOR=OFF
)
endif()

link_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/pugixml_build
    ${CMAKE_CURRENT_BINARY_DIR}/spglib_build
    ${CMAKE_CURRENT_BINARY_DIR}/scalapack_build/lib
)

include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/pugixml_src/src
    ${CMAKE_CURRENT_BINARY_DIR}/spglib_src/src
    ${CMAKE_CURRENT_BINARY_DIR}/eigen_src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest/googletest/include
    ${CMAKE_CURRENT_BINARY_DIR}/json_src/single_include
    ${CMAKE_CURRENT_BINARY_DIR}/highfive_src/include
)

