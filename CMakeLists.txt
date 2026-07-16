CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(RpolyPlusPlus C CXX)

# The library is written against C++17 (constexpr static members,
# [[nodiscard]], implicitly-inline static constexpr initializers).
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build; this library is intended for hot loops.
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF ()
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

# Let Homebrew-installed dependencies (Eigen, GoogleTest) be discovered without
# a manual -DCMAKE_PREFIX_PATH, including on Apple Silicon (/opt/homebrew).
IF (APPLE)
  EXECUTE_PROCESS(COMMAND brew --prefix
    OUTPUT_VARIABLE HOMEBREW_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  IF (HOMEBREW_PREFIX)
    LIST(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")
  ENDIF ()
ENDIF ()

# By default, the unit tests will be built.
OPTION(BUILD_TESTING "Enable testing" ON)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Modern Eigen ships its own CMake config package, which exposes the
# Eigen3::Eigen imported target. The target carries Eigen's include directories
# as SYSTEM includes, so Eigen's own compiler warnings are suppressed for us
# (this replaces the old vendored FindEigen.cmake module).
#
# The code is compatible with Eigen 3.4 through 5.x. We intentionally do not
# pin a version here: Eigen's config uses SameMajorVersion compatibility, so a
# request like "3.4" would be rejected against an installed Eigen 5.x.
FIND_PACKAGE(Eigen3 REQUIRED NO_MODULE)
GET_TARGET_PROPERTY(EIGEN_INCLUDE_DIRS Eigen3::Eigen
  INTERFACE_INCLUDE_DIRECTORIES)
MESSAGE("-- Found Eigen version ${Eigen3_VERSION}: ${EIGEN_INCLUDE_DIRS}")

# Use a larger inlining threshold for Clang, since it hobbles Eigen,
# resulting in an unreasonably slow version of the blas routines. The
# -Qunused-arguments is needed because CMake passes the inline
# threshold to the linker and clang complains about it and dies.
IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Qunused-arguments -mllvm -inline-threshold=600")
  # Older versions of Clang (<= 2.9) do not support the 'return-type-c-linkage'
  # option, so check for its presence before adding it to the default flags set.
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-Wno-return-type-c-linkage"
    HAVE_RETURN_TYPE_C_LINKAGE)
  IF (HAVE_RETURN_TYPE_C_LINKAGE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type-c-linkage")
  ENDIF(HAVE_RETURN_TYPE_C_LINKAGE)
ENDIF ()

SET(RPOLY_PLUS_PLUS_SRCS
  src/polynomial.cc
  src/find_polynomial_roots_jenkins_traub.cc)

ADD_LIBRARY(rpoly_plus_plus SHARED ${RPOLY_PLUS_PLUS_SRCS})
TARGET_LINK_LIBRARIES(rpoly_plus_plus PUBLIC Eigen3::Eigen)

# Optional accuracy/speed evaluation harness (off by default).
OPTION(BUILD_BENCHMARK "Build the accuracy/speed benchmark" OFF)
IF (BUILD_BENCHMARK)
  ADD_EXECUTABLE(rpoly_benchmark benchmark/rpoly_benchmark.cc)
  TARGET_LINK_LIBRARIES(rpoly_benchmark PRIVATE rpoly_plus_plus Eigen3::Eigen)

  # Optional baseline solver: the original RPOLY (Jenkins-Traub reference). Drop
  # David Binner's RPOLY_AK1 C++ port at third_party/rpoly/rpoly_ak1.cpp (with
  # its demo main() removed) -- see third_party/rpoly/README.md.
  OPTION(WITH_RPOLY "Compare against RPOLY (third_party/rpoly/rpoly_ak1.cpp)" OFF)
  IF (WITH_RPOLY)
    IF (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rpoly/rpoly_ak1.cpp")
      MESSAGE(FATAL_ERROR
        "WITH_RPOLY=ON but third_party/rpoly/rpoly_ak1.cpp is missing. "
        "See third_party/rpoly/README.md.")
    ENDIF ()
    TARGET_SOURCES(rpoly_benchmark PRIVATE third_party/rpoly/rpoly_ak1.cpp)
    TARGET_COMPILE_DEFINITIONS(rpoly_benchmark PRIVATE WITH_RPOLY)
    # RPOLY_AK1 ships with a demo int main(); rename it at compile time so it
    # does not collide with the benchmark's main. This means the vendored file
    # can be dropped in verbatim -- no need to hand-edit out its main().
    SET_SOURCE_FILES_PROPERTIES(third_party/rpoly/rpoly_ak1.cpp PROPERTIES
      COMPILE_DEFINITIONS "main=rpoly_ak1_disabled_main")
  ENDIF ()

  # Optional high-accuracy oracle: MPSolve (linked, not vendored; GPL). See
  # third_party/mpsolve/README.md for building libmps on top of GMP.
  OPTION(WITH_MPSOLVE "Compare against MPSolve as a high-accuracy oracle" OFF)
  IF (WITH_MPSOLVE)
    SET(MPSOLVE_ROOT "" CACHE PATH
      "Install prefix of MPSolve (contains include/mps and lib/libmps)")
    FIND_PATH(MPS_INCLUDE_DIR mps/mps.h HINTS "${MPSOLVE_ROOT}/include")
    FIND_LIBRARY(MPS_LIBRARY NAMES mps HINTS "${MPSOLVE_ROOT}/lib")
    FIND_LIBRARY(GMP_LIBRARY NAMES gmp HINTS "${MPSOLVE_ROOT}/lib")
    IF (NOT MPS_INCLUDE_DIR OR NOT MPS_LIBRARY OR NOT GMP_LIBRARY)
      MESSAGE(FATAL_ERROR
        "WITH_MPSOLVE=ON but libmps/GMP were not found. "
        "Pass -DMPSOLVE_ROOT=<prefix>; see third_party/mpsolve/README.md.")
    ENDIF ()
    TARGET_INCLUDE_DIRECTORIES(rpoly_benchmark PRIVATE "${MPS_INCLUDE_DIR}")
    TARGET_LINK_LIBRARIES(rpoly_benchmark PRIVATE "${MPS_LIBRARY}" "${GMP_LIBRARY}")
    TARGET_COMPILE_DEFINITIONS(rpoly_benchmark PRIVATE WITH_MPSOLVE)
  ENDIF ()

  # Heap-free probe variant: compile the benchmark together with the solver
  # sources so Eigen's runtime allocation gate is active in every translation
  # unit (the solver's own allocations included). The gate is portable -- it
  # uses EIGEN_RUNTIME_NO_MALLOC plus a throwing eigen_assert
  # (benchmark/eigen_malloc_gate.h) rather than glibc malloc interposition --
  # so the heap column is populated on macOS as well as Linux.
  ADD_EXECUTABLE(rpoly_benchmark_heapcheck
    benchmark/rpoly_benchmark.cc
    ${RPOLY_PLUS_PLUS_SRCS})
  TARGET_COMPILE_DEFINITIONS(rpoly_benchmark_heapcheck
    PRIVATE RPOLY_BENCH_NO_MALLOC_GATE)
  # Force-include the gate header ahead of every source (including the solver
  # translation units) so EIGEN_RUNTIME_NO_MALLOC is defined before any Eigen
  # header is seen. -include is supported by GCC and Clang.
  IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    TARGET_COMPILE_OPTIONS(rpoly_benchmark_heapcheck PRIVATE
      -include "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/eigen_malloc_gate.h")
  ENDIF ()
  TARGET_LINK_LIBRARIES(rpoly_benchmark_heapcheck PRIVATE Eigen3::Eigen)
ENDIF ()

IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
  MESSAGE("\n=================================================================================")
  MESSAGE("\n-- Build type: Debug. Performance will be terrible!")
  MESSAGE("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
  MESSAGE("\n=================================================================================")
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug")

IF (BUILD_TESTING)
  # GoogleTest: prefer a system/Homebrew install if present, otherwise fetch a
  # pinned modern release. GoogleTest 1.17 targets C++17, matching this library.
  FIND_PACKAGE(GTest QUIET)
  IF (NOT GTest_FOUND)
    INCLUDE(FetchContent)
    # Set extracted-file timestamps to extraction time (CMake >= 3.24) so a
    # changed download URL correctly triggers a rebuild.
    IF (POLICY CMP0135)
      CMAKE_POLICY(SET CMP0135 NEW)
    ENDIF ()
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
      URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
    # Do not add GoogleTest to this project's install set.
    SET(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  ENDIF ()

  ENABLE_TESTING()

  MACRO (GTEST FILENAME)
    STRING(REPLACE "/" ";" TEST_NAME_LIST ${FILENAME})
    LIST(REVERSE TEST_NAME_LIST)
    LIST(GET TEST_NAME_LIST 0 TEST_NAME)
    ADD_EXECUTABLE(${TEST_NAME}_test test/test_main.cc test/${FILENAME}_test.cc)
    TARGET_LINK_LIBRARIES(${TEST_NAME}_test
      GTest::gtest
      rpoly_plus_plus)
    ADD_TEST(NAME ${TEST_NAME}_test
      COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_NAME}_test)
  ENDMACRO (GTEST)

  GTEST(find_polynomial_roots_jenkins_traub)
ENDIF (BUILD_TESTING)
