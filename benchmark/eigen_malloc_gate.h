// Portable heap-allocation gate for the benchmark's heap-free probe.
//
// This must be seen before any Eigen header, in every translation unit whose
// allocations we want to observe -- the benchmark AND the solver sources when
// they are compiled together into the rpoly_benchmark_heapcheck target (the
// build system force-includes this header into those sources with -include).
//
// Defining EIGEN_RUNTIME_NO_MALLOC makes Eigen route every dynamic allocation
// through check_that_malloc_is_allowed(); the throwing eigen_assert below turns
// a disallowed allocation into a catchable C++ exception instead of an abort().
// A probe can then wrap a single solve in set_is_malloc_allowed(false) and
// catch to detect whether that solve touched the heap.
//
// This replaces the original benchmark's glibc __libc_malloc interposition,
// which does not exist on macOS / non-glibc platforms.
#ifndef RPOLY_BENCHMARK_EIGEN_MALLOC_GATE_H_
#define RPOLY_BENCHMARK_EIGEN_MALLOC_GATE_H_

#ifndef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_NO_MALLOC
#endif

#include <stdexcept>

#ifndef eigen_assert
#define eigen_assert(X)                                       \
  do {                                                        \
    if (!(X)) throw std::runtime_error("eigen: disallowed"); \
  } while (0)
#endif

#endif  // RPOLY_BENCHMARK_EIGEN_MALLOC_GATE_H_
