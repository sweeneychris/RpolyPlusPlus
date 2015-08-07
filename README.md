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

Dependencies
------------

Eigen3 library: http://eigen.tuxfamily.org/

This library is header-only so the installation is simple.

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
default. To build without the unit tests change the cmake line to:

```
cmake .. -DBUILD_TESTING=Off
```

If testing is enabled, you can run the unit test from the build directory with:
```
./bin/find_polynomial_roots_jenkins_traub_test
```

All unit tests should pass.

Questions
---------

Contact Chris Sweeney at sweeney.chris.m@gmail.com
