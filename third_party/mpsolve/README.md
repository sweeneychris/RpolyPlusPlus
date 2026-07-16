# MPSolve (Bini & Robol) — high-accuracy oracle for the benchmark

The benchmark can use MPSolve as a multiprecision ground-truth oracle for root
positions (forward-error reference on families with no analytic closed form).
**No third-party code is committed here** — you build/install MPSolve yourself.

  https://github.com/robol/MPSolve

GMP and MPFR (its dependencies) are already available via Homebrew.

## Build and install

```
git clone https://github.com/robol/MPSolve.git
cd MPSolve
./autogen.sh
./configure --prefix=/opt/homebrew           # or another prefix on your PATH
make -j
make install
```

(Autotools + GMP/MPFR must be present: `brew install autoconf automake libtool gmp mpfr`.)

## Enable in the benchmark

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARK=ON \
      -DWITH_MPSOLVE=ON -DMPSOLVE_ROOT=/opt/homebrew
cmake --build build --target rpoly_benchmark
```

`MPSOLVE_ROOT` is only needed if `mps/mps.h` / `libmps` are not on the default
search paths. The adapter (`benchmark/solvers/mpsolve_adapter.h`) uses the C
API: `mps_context_new` → `mps_monomial_poly_*` → `mps_mpsolve` →
`mps_context_get_roots_d`. If your MPSolve version renames any of these,
adjust the adapter.

## License note

MPSolve is GPL-licensed. Because it is linked (not vendored) and only into the
optional `rpoly_benchmark` executable, it does not affect the license of the
`rpoly_plus_plus` library itself. Do not distribute a combined binary without
honoring the GPL.
