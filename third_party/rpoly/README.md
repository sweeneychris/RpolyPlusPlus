# RPOLY (Jenkins & Traub, ACM TOMS Algorithm 493) — benchmark baseline

The benchmark can compare against the original RPOLY algorithm. This directory
is the drop-in slot for its source; **no third-party code is committed here.**

## Recommended: RPOLY_AK1 (self-contained C++ port)

David Binner's RPOLY_AK1 is a single-file C++ translation of RPOLY that compiles
with any C++ compiler (no Fortran toolchain needed):

  https://www.akiti.ca/rpoly_ak1_cpp.html

1. Save the source as `third_party/rpoly/rpoly_ak1.cpp` (verbatim is fine — the
   build renames its demo `int main()` automatically via a per-source
   `-Dmain=...`, so there is no duplicate-symbol clash and nothing to hand-edit).
2. Configure with the option enabled:

   ```
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARK=ON -DWITH_RPOLY=ON
   cmake --build build --target rpoly_benchmark
   ```

The adapter (`benchmark/solvers/rpoly_ak1_adapter.h`) expects the signature

    void rpoly_ak1(double op[], int* Degree, double zeror[], double zeroi[]);

with the highest-degree coefficient first and `MAXDEGREE == 100`. If the
revision you download differs, adjust the declaration/success-test there.

## Alternative: the original Fortran (TOMS 493)

The reference Fortran `rpoly.f` is on Netlib: https://netlib.org/toms/493
It is under the **ACM Software License Agreement** (redistribution restricted),
so it is intentionally not vendored. To use it:

1. `brew install gcc` (provides `gfortran`).
2. Place `rpoly.f` here and add a C shim; enable Fortran in CMake
   (`enable_language(Fortran)`) and wire it the same way as the AK1 path.

## License note

Whatever you place here keeps its own license. RPOLY_AK1 is distributed freely
by its author; the TOMS 493 Fortran is ACM-licensed. Neither is redistributed
as part of this repository.
