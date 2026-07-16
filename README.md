RPOLY - A Polynomial Root-finding library
=========================================

A three-stage algorithm for finding roots of polynomials with real coefficients
as outlined in: "A Three-Stage Algorithm for Real Polynomials Using Quadratic
Iteration" by Jenkins and Traub, SIAM 1970. Please note that this variant is
different than the complex-coefficient version, and is estimated to be up to 4
times faster.

The algorithm works by computing shifts in so-called "K-polynomials" that reveal
the roots. These shifts are applied in three stages: Zero-shifts, Fixed-shifts,
and Variable-shift iterations. Roots are revealed as real roots or as a pair of
complex conjugate roots. After a root (or pair of roots) is found, it is divided
from the polynomial and the process is repeated.

Usage
-----

The library exposes a single function, declared in
`src/find_polynomial_roots_jenkins_traub.h`:

```cpp
namespace rpoly_plus_plus {

// Finds the roots of a real-coefficient polynomial. Coefficients are supplied
// highest-degree first: `polynomial(0)` multiplies x^n and the last entry is
// the constant term. On success the real and imaginary parts of each root are
// written to `real_roots` and `complex_roots` respectively (complex-conjugate
// pairs occupy consecutive entries). Either output pointer may be null if you
// don't need that part. Returns false if the roots could not be extracted, so
// callers must check the result.
[[nodiscard]] bool FindPolynomialRootsJenkinsTraub(
    const Eigen::VectorXd& polynomial,
    Eigen::VectorXd* real_roots,
    Eigen::VectorXd* complex_roots);

}  // namespace rpoly_plus_plus
```

Minimal example — find the roots of `x^2 - 3x + 2` (which are 1 and 2):

```cpp
#include <iostream>

#include <Eigen/Core>

#include "src/find_polynomial_roots_jenkins_traub.h"

int main() {
  // Coefficients, highest degree first: x^2 - 3x + 2.
  Eigen::VectorXd coeffs(3);
  coeffs << 1.0, -3.0, 2.0;

  Eigen::VectorXd real_roots, imag_roots;
  if (!rpoly_plus_plus::FindPolynomialRootsJenkinsTraub(coeffs, &real_roots,
                                                        &imag_roots)) {
    std::cerr << "Root finding failed.\n";
    return 1;
  }

  for (int i = 0; i < real_roots.size(); ++i) {
    std::cout << real_roots(i) << " + " << imag_roots(i) << "i\n";
  }
  return 0;
}
```

The CMake build produces a `rpoly_plus_plus` library target that carries its
Eigen include paths. To consume it from another CMake project, add this
repository as a subdirectory (there is no `install()` / `find_package` export
yet) and link the target:

```cmake
add_subdirectory(RpolyPlusPlus)
target_link_libraries(your_target PRIVATE rpoly_plus_plus)
```

Without CMake, compile your program together with the two library sources (run
from the repository root and point `-I` at your Eigen include directory):

```sh
g++ -std=c++17 -O2 -I. -I/opt/homebrew/include/eigen3 \
    your_program.cc \
    src/find_polynomial_roots_jenkins_traub.cc src/polynomial.cc \
    -o your_program
```

`FindPolynomialRootsJenkinsTraub` returns `false` when it cannot extract the
roots (for example, an empty coefficient vector); a constant (degree-0)
polynomial returns `true` with no roots. Because the function is
`[[nodiscard]]`, always check the returned value.

Accuracy and speed
------------------

The repository ships an optional benchmark
(`cmake .. -DBUILD_BENCHMARK=ON && make rpoly_benchmark && ./bin/rpoly_benchmark`)
that evaluates the solver on literature-standard numerical families (Wilkinson,
Mignotte, cyclotomic, Kac, multiplicity ladders, wide dynamic range) and on
computer-vision / geometric families with verifiable ground truth (P3P, 7-point
fundamental, ray-torus, point-to-ellipse), each with condition-aware pass/fail
gates. It also compares against other solvers — a companion-matrix eigenvalue
solver (Eigen), a from-scratch Aberth–Ehrlich iteration, and, when their sources
are supplied, the original RPOLY (TOMS 493) and MPSolve (see `third_party/`).

> **The absolute timings below are from a single machine** (Apple M4 Max,
> macOS 26, Apple clang 17, `-O3`) and are indicative only. The **ratios** are
> the portable part; absolute numbers will differ on your hardware. Reproduce
> with the command above.

**Accuracy (hardware-independent).** The key metric is the *normalized backward
residual* `|P(z)| / Σ|aᵢ||z|ⁿ⁻ⁱ`: a returned root is backward-stable when this is
`O(ε)`, no matter how ill-conditioned the polynomial is. Worst case over all
benchmark families:

| solver | max backward residual | notes |
|---|---|---|
| **RpolyPlusPlus** | **≈5e-16** (≈ 1 ε) | backward-stable on every family tested |
| Original RPOLY (TOMS 493) | ≈3e-13 | |
| Aberth–Ehrlich (reference impl.) | ≈9e-16 | |
| Companion-matrix QR (Eigen) | ≈5e-7 | not backward-stable on wide dynamic range |

Forward accuracy (root *positions*) is condition-limited — near machine
precision for well-separated roots, and about `ε^(1/m)` for a root of
multiplicity `m`, which is the best any double-precision solver can do.

**Speed (Apple M4 Max, random real roots).** Median solve time by degree:

| degree | RpolyPlusPlus | companion-matrix QR |
|---|---|---|
| 4  | ~0.7 µs | ~1.6 µs |
| 8  | ~2.4 µs | ~6.1 µs |
| 16 | ~8 µs   | ~28 µs  |
| 32 | ~30 µs  | ~149 µs |
| 64 | ~229 µs | ~975 µs |

Across a mixed corpus this is roughly **3–5× faster than the companion-matrix
eigenvalue method** (the margin grows with degree), and about **1.9× the time of
the bare original RPOLY** — the difference being the root polishing and
backward-error acceptance gate that yield the ~100–1000× smaller residuals on
ill-conditioned inputs shown above.

The benchmark's fourth solver, a from-scratch Aberth–Ehrlich iteration, is
included only as an *accuracy* baseline (it reaches the same machine-level
residuals through a different algorithm class). Its timing is intentionally left
out of the comparison above: it is a simple, unoptimized reference
implementation, so its numbers would misrepresent the speed of a tuned
Aberth–Ehrlich solver rather than say anything meaningful about this library.

**Allocation.** For polynomials of degree ≤ 32, a solve performs **zero heap
allocations** (it is entirely stack-resident); this is verified by the
`rpoly_benchmark_heapcheck` build target.

Dependencies
------------

* A C++17 compiler.
* CMake >= 3.16.
* Eigen (>= 3.4), consumed via its own CMake config package
  (http://eigen.tuxfamily.org/). e.g. `brew install eigen` or
  `apt-get install libeigen3-dev`.

The unit tests use GoogleTest. If a copy is already installed it is used;
otherwise a pinned release is downloaded automatically at configure time via
CMake's FetchContent (this requires network access on the first configure).
On macOS the Homebrew prefix (including Apple Silicon's `/opt/homebrew`) is
detected automatically, so no `-DCMAKE_PREFIX_PATH` is needed.

Building
--------
Run the following commands from the root directory of RpolyPlusPlus.

```
mkdir build
cd build
cmake ..
make
```

This should build the library. Note that the unit tests are enabled by
default. To build without the unit tests (and skip the GoogleTest dependency)
change the cmake line to:

```
cmake .. -DBUILD_TESTING=Off
```

If testing is enabled, you can run the unit test from the build directory with:
```
./bin/find_polynomial_roots_jenkins_traub_test
```

or through CTest:
```
ctest --output-on-failure
```

All unit tests should pass.

Questions
---------

Contact Chris Sweeney at sweeney.chris.m@gmail.com
