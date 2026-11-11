// Copyright (C) 2015 Chris Sweeney
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Chris Sweeney nor the names of its contributors may
//       be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.

// Accuracy tests across root magnitudes and coefficient ranges.
//
// The solver contains absolute tolerances, and its exact power-of-two
// rescaling only engages when the geometric mean of the roots is beyond
// 2^+-32. Roots in between (roughly 1e-9..1e-3 and 1e3..1e9) are solved
// unscaled, so an absolute threshold that is right at scale 1 can be badly
// wrong there. Example: a linear-shift early exit accepting any iterate with
// |P(s)| <= 1e-14 accepted essentially arbitrary points for roots ~1e-5
// (where |P| is ~1e-15 throughout the cluster) and silently returned wrong
// roots. These tests check that relative accuracy does not depend on scale.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "src/find_polynomial_roots_jenkins_traub.h"

namespace rpoly_plus_plus {
namespace {

using Eigen::VectorXd;
using Complex = std::complex<double>;

// Monic polynomial with the given roots (conjugates included), highest-degree
// coefficient first.
VectorXd FromRoots(const std::vector<Complex>& roots) {
  std::vector<Complex> p{1.0};
  for (const Complex& r : roots) {
    std::vector<Complex> q(p.size() + 1, 0.0);
    for (size_t i = 0; i < p.size(); ++i) {
      q[i] += p[i];
      q[i + 1] -= r * p[i];
    }
    p = q;
  }
  VectorXd poly(p.size());
  for (size_t i = 0; i < p.size(); ++i) poly(i) = p[i].real();
  return poly;
}

// Solves `poly` and returns the worst relative error |z - r| / |r| over the
// true roots, matching each true root greedily to its nearest unused computed
// root. Returns +inf if the solver fails or returns the wrong root count.
double MaxRelativeError(const VectorXd& poly,
                        const std::vector<Complex>& truth) {
  VectorXd re, im;
  if (!FindPolynomialRootsJenkinsTraub(poly, &re, &im) ||
      re.size() != static_cast<Eigen::Index>(truth.size())) {
    return HUGE_VAL;
  }
  std::vector<bool> used(re.size(), false);
  double worst = 0.0;
  for (const Complex& r : truth) {
    Eigen::Index best = -1;
    double best_dist = HUGE_VAL;
    for (Eigen::Index i = 0; i < re.size(); ++i) {
      const double d = std::abs(Complex(re(i), im(i)) - r);
      if (!used[i] && (best < 0 || d < best_dist)) {
        best = i;
        best_dist = d;
      }
    }
    used[best] = true;
    const double e = best_dist / std::abs(r);
    worst = std::isnan(e) ? HUGE_VAL : std::max(worst, e);
  }
  return worst;
}

// Well-conditioned random root pattern of the given degree: real roots and
// conjugate pairs with magnitudes in ~[0.1, 1.5], no root near 0.
std::vector<Complex> RandomUnitPattern(int degree, std::mt19937* rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<Complex> roots;
  while (static_cast<int>(roots.size()) < degree) {
    if (degree - static_cast<int>(roots.size()) >= 2 && ((*rng)() & 1)) {
      const Complex z(u(*rng), 0.1 + std::abs(u(*rng)));
      roots.push_back(z);
      roots.push_back(std::conj(z));
    } else {
      double x = u(*rng);
      if (std::abs(x) < 0.1) x += x < 0 ? -0.2 : 0.2;
      roots.push_back(x);
    }
  }
  return roots;
}

// For these well-conditioned patterns the solver reaches ~1e-11 relative
// error at every scale; 1e-8 leaves ample margin across compilers.
constexpr double kRelTol = 1e-8;

// Reports at most a few failures per test so output stays readable.
class FailureBudget {
 public:
  bool Take() { return ++count_ <= 5; }
  int count() const { return count_; }

 private:
  int count_ = 0;
};

void ExpectAccurate(const std::vector<Complex>& roots, const char* what,
                    double scale, FailureBudget* budget) {
  const VectorXd poly = FromRoots(roots);
  const double err = MaxRelativeError(poly, roots);
  if (!(err <= kRelTol) && budget->Take()) {
    ADD_FAILURE() << what << " at scale " << scale
                  << ": max relative root error " << err
                  << "\n  poly = " << poly.transpose();
  }
}

// ---------------------------------------------------------------------------

// The readable regression: three simple roots near 1e-5 came back as
// {3.17e-5, 1.42e-5 +- 4.27e-6 i} while reporting success.
TEST(ScaleRobustness, SmallRootsRegression) {
  FailureBudget budget;
  ExpectAccurate({1e-5, 2e-5, 3e-5}, "roots {1,2,3}e-5", 1e-5, &budget);
  ExpectAccurate({1e-3, -2e-3, Complex(1e-3, 3e-3), Complex(1e-3, -3e-3)},
                 "mixed roots ~1e-3", 1e-3, &budget);
}

// roots(p(x / s)) = s * roots(p): relative accuracy must not depend on s.
// Non-powers of two so the scaling is not exact in binary.
TEST(ScaleRobustness, RootScaleInvariance) {
  std::mt19937 rng(20261110);
  std::vector<std::vector<Complex>> patterns;
  for (int t = 0; t < 120; ++t) {
    patterns.push_back(RandomUnitPattern(3 + t % 8, &rng));  // degree 3-10
  }
  FailureBudget budget;
  for (int e = -12; e <= 12; ++e) {
    const double s = 1.37 * std::pow(10.0, e);
    for (const auto& pattern : patterns) {
      std::vector<Complex> roots;
      for (const Complex& z : pattern) roots.push_back(s * z);
      ExpectAccurate(roots, "random pattern", s, &budget);
    }
  }
  EXPECT_EQ(budget.count(), 0) << "inaccurate solves";
}

// A few tiny roots next to roots of size ~1. The tiny roots are determined by
// the trailing coefficients to full relative precision, so they must be found
// to full relative accuracy, not merely to an absolute 1e-14.
TEST(ScaleRobustness, SmallRootsBesideUnitRoots) {
  std::mt19937 rng(7);
  FailureBudget budget;
  for (int e = 3; e <= 13; ++e) {
    const double delta = std::pow(10.0, -e);
    for (int t = 0; t < 60; ++t) {
      std::vector<Complex> roots = RandomUnitPattern(3 + t % 4, &rng);
      const double sign = (t & 1) ? 1.0 : -1.0;
      for (int j = 0; j < 1 + t % 3; ++j) {
        roots.push_back(sign * delta * (1.0 + 0.7 * j));
      }
      ExpectAccurate(roots, "tiny roots beside unit roots", delta, &budget);
    }
  }
  EXPECT_EQ(budget.count(), 0) << "inaccurate solves";
}

// The mirror image: a few huge roots next to roots of size ~1.
TEST(ScaleRobustness, LargeRootsBesideUnitRoots) {
  std::mt19937 rng(8);
  FailureBudget budget;
  for (int e = 3; e <= 13; ++e) {
    const double big = std::pow(10.0, e);
    for (int t = 0; t < 60; ++t) {
      std::vector<Complex> roots = RandomUnitPattern(3 + t % 4, &rng);
      const double sign = (t & 1) ? 1.0 : -1.0;
      for (int j = 0; j < 1 + t % 3; ++j) {
        roots.push_back(sign * big * (1.0 + 0.7 * j));
      }
      ExpectAccurate(roots, "huge roots beside unit roots", big, &budget);
    }
  }
  EXPECT_EQ(budget.count(), 0) << "inaccurate solves";
}

// ---------------------------------------------------------------------------
// Closed-form quadratic. It also runs inside the iteration (roots of each
// quadratic factor) and on the final deflated quadratic.
// ---------------------------------------------------------------------------

struct QuadraticCase {
  double b, c;  // x^2 + b x + c
  Complex r1, r2;
};

TEST(ScaleRobustness, QuadraticWithExtremeCoefficients) {
  const QuadraticCase cases[] = {
      // b*b overflows (|b| > ~1.3e154); roots ~ -b and -c/b.
      {1e200, 1.0, -1e200, -1e-200},
      {-1e200, 1.0, 1e200, 1e-200},
      {1e160, -1.0, -1e160, 1e-160},
      {1e300, 1e300, -1e300, -1.0},
      // 4*c overflows; complex pair -1/2 +- ~1e154 i.
      {1.0, 1e308, Complex(-0.5, std::sqrt(1e308 - 0.25)),
       Complex(-0.5, -std::sqrt(1e308 - 0.25))},
      // Controls: ordinary magnitudes.
      {-3.0, 2.0, 1.0, 2.0},
      {2.0, 5.0, Complex(-1.0, 2.0), Complex(-1.0, -2.0)},
  };
  for (const QuadraticCase& q : cases) {
    VectorXd poly(3);
    poly << 1.0, q.b, q.c;
    EXPECT_LE(MaxRelativeError(poly, {q.r1, q.r2}), 1e-12)
        << "x^2 + " << q.b << " x + " << q.c;
  }
}

// Roots spanning many orders of magnitude with geometric mean ~1, so the
// power-of-two rescaling does not engage; intermediate quantities (the
// quadratic factor's discriminant) must not overflow.
// Known limit: at spreads ~1e+-150 the coefficients reach ~1e225 and products
// in the K-polynomial update can overflow (FMA-dependent), so stop at 1e+-100.
TEST(ScaleRobustness, WideDynamicRange) {
  for (const double k : {5.0, 20.0, 50.0, 100.0}) {
    const double lo = std::pow(10.0, -k), hi = std::pow(10.0, k);
    const std::vector<Complex> r4 = {lo, hi, 1.5, -2.5};
    EXPECT_LE(MaxRelativeError(FromRoots(r4), r4), 1e-12) << "spread 1e" << k;
    const std::vector<Complex> r6 = {lo,  -std::pow(10.0, -k / 2),
                                     std::pow(10.0, k / 2), hi,
                                     Complex(0.3, 1.1), Complex(0.3, -1.1)};
    EXPECT_LE(MaxRelativeError(FromRoots(r6), r6), 1e-12) << "spread 1e" << k;
  }
}

// A nearly vanishing leading coefficient, as produced by near-degenerate
// geometry: (eps x + 1)(x - 1)(x - 2)(x - 3) has one root at -1/eps.
TEST(ScaleRobustness, TinyLeadingCoefficient) {
  const VectorXd cubic = FromRoots({1.0, 2.0, 3.0});
  for (const double eps : {1e-8, 1e-16, 1e-30, 1e-100, 1e-300}) {
    VectorXd poly = VectorXd::Zero(5);
    poly.head(4) += eps * cubic;
    poly.tail(4) += cubic;
    EXPECT_LE(MaxRelativeError(poly, {1.0, 2.0, 3.0, -1.0 / eps}), 1e-12)
        << "leading coefficient " << eps;
  }
}

}  // namespace
}  // namespace rpoly_plus_plus
