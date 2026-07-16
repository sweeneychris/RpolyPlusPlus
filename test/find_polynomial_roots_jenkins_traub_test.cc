// Copyright (C) 2015 Chris Sweeney. All rights reserved.
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
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include <algorithm>
#include <complex>
#include <limits>
#include <random>
#include <vector>
#include "gtest/gtest.h"

#include "src/find_polynomial_roots_jenkins_traub.h"
#include "src/polynomial.h"
#include "test/test_utils.h"

namespace rpoly_plus_plus {

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// For IEEE-754 doubles, machine precision is about 2e-16.
const double kEpsilon = 1e-10;
const double kEpsilonLoose = 1e-8;

// Return the constant polynomial p(x) = 1.23.
VectorXd ConstantPolynomial(double value) {
  VectorXd poly(1);
  poly(0) = value;
  return poly;
}

// Return the polynomial p(x) = poly(x) * (x - root).
VectorXd AddRealRoot(const VectorXd& poly, double root) {
  VectorXd poly2(poly.size() + 1);
  poly2.setZero();
  poly2.head(poly.size()) += poly;
  poly2.tail(poly.size()) -= root * poly;
  return poly2;
}

// Return the polynomial
// p(x) = poly(x) * (x - real - imag*i) * (x - real + imag*i).
VectorXd AddComplexRootPair(const VectorXd& poly, double real,
                            double imag) {
  VectorXd poly2(poly.size() + 2);
  poly2.setZero();
  // Multiply poly by x^2 - 2real + abs(real,imag)^2
  poly2.head(poly.size()) += poly;
  poly2.segment(1, poly.size()) -= 2 * real * poly;
  poly2.tail(poly.size()) += (real * real + imag * imag) * poly;
  return poly2;
}

// Sort the entries in a vector.
// Needed because the roots are not returned in sorted order.
VectorXd SortVector(const VectorXd& in) {
  VectorXd out(in);
  std::sort(out.data(), out.data() + out.size());
  return out;
}

// Run a test with the polynomial defined by the N real roots in roots_real.
// If use_real is false, NULL is passed as the real argument to
// FindPolynomialRoots. If use_imaginary is false, NULL is passed as the
// imaginary argument to FindPolynomialRoots.
template <int N>
void RunPolynomialTestRealRoots(const double (&real_roots)[N], bool use_real,
                                bool use_imaginary, double epsilon) {
  VectorXd real;
  VectorXd imaginary;
  VectorXd poly = ConstantPolynomial(1.23);
  for (int i = 0; i < N; ++i) {
    poly = AddRealRoot(poly, real_roots[i]);
  }
  VectorXd* const real_ptr = use_real ? &real : NULL;
  VectorXd* const imaginary_ptr = use_imaginary ? &imaginary : NULL;
  bool success = FindPolynomialRootsJenkinsTraub(poly, real_ptr, imaginary_ptr);

  EXPECT_EQ(success, true);
  if (use_real) {
    EXPECT_EQ(real.size(), N);
    real = SortVector(real);
    test::ExpectArraysNear(N, real.data(), real_roots, epsilon);
  }
  if (use_imaginary) {
    EXPECT_EQ(imaginary.size(), N);
    const VectorXd zeros = VectorXd::Zero(N);
    test::ExpectArraysNear(N, imaginary.data(), zeros.data(), epsilon);
  }
}

}  // namespace

TEST(Polynomial, InvalidPolynomialOfZeroLengthIsRejected) {
  // Vector poly(0) is an ambiguous constructor call, so
  // use the constructor with explicit column count.
  VectorXd poly(0, 1);
  VectorXd real;
  VectorXd imag;
  bool success = FindPolynomialRootsJenkinsTraub(poly, &real, &imag);

  EXPECT_EQ(success, false);
}

TEST(Polynomial, ConstantPolynomialReturnsNoRoots) {
  VectorXd poly = ConstantPolynomial(1.23);
  VectorXd real;
  VectorXd imag;
  bool success = FindPolynomialRootsJenkinsTraub(poly, &real, &imag);

  EXPECT_EQ(success, true);
  EXPECT_EQ(real.size(), 0);
  EXPECT_EQ(imag.size(), 0);
}

TEST(Polynomial, LinearPolynomialWithPositiveRootWorks) {
  const double roots[1] = { 42.42 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, LinearPolynomialWithNegativeRootWorks) {
  const double roots[1] = { -42.42 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuadraticPolynomialWithPositiveRootsWorks) {
  const double roots[2] = { 1.0, 42.42 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuadraticPolynomialWithOneNegativeRootWorks) {
  const double roots[2] = { -42.42, 1.0 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuadraticPolynomialWithTwoNegativeRootsWorks) {
  const double roots[2] = { -42.42, -1.0 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuadraticPolynomialWithCloseRootsWorks) {
  const double roots[2] = { 42.42, 42.43 };
  RunPolynomialTestRealRoots(roots, true, false, kEpsilonLoose);
}

TEST(Polynomial, QuadraticPolynomialWithComplexRootsWorks) {
  VectorXd real;
  VectorXd imag;

  VectorXd poly = ConstantPolynomial(1.23);
  poly = AddComplexRootPair(poly, 42.42, 4.2);
  bool success = FindPolynomialRootsJenkinsTraub(poly, &real, &imag);

  EXPECT_EQ(success, true);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  EXPECT_NEAR(real(0), 42.42, kEpsilon);
  EXPECT_NEAR(real(1), 42.42, kEpsilon);
  EXPECT_NEAR(std::abs(imag(0)), 4.2, kEpsilon);
  EXPECT_NEAR(std::abs(imag(1)), 4.2, kEpsilon);
  EXPECT_NEAR(std::abs(imag(0) + imag(1)), 0.0, kEpsilon);
}

TEST(Polynomial, QuarticPolynomialWorks) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilonLoose);
}

TEST(Polynomial, QuarticPolynomialWithTwoClustersOfCloseRootsWorks) {
  const double roots[4] = { 1.23e-1, 2.46e-1, 1.23e+5, 2.46e+5 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilonLoose);
}

TEST(Polynomial, QuarticPolynomialWithTwoZeroRootsWorks) {
  const double roots[4] = { -42.42, 0.0, 0.0, 42.42 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilonLoose);
}

TEST(Polynomial, QuarticMonomialWorks) {
  const double roots[4] = { 0.0, 0.0, 0.0, 0.0 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, NullPointerAsImaginaryPartWorks) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, true, false, kEpsilonLoose);
}

TEST(Polynomial, NullPointerAsRealPartWorks) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, false, true, kEpsilon);
}

TEST(Polynomial, BothOutputArgumentsNullWorks) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, false, false, kEpsilon);
}

TEST(Polynomial, JenkinsTraubManyRoots) {
  static const int N = 25;
  VectorXd poly = ConstantPolynomial(1.23);
  VectorXd roots = VectorXd::Random(N);
  roots = SortVector(roots);

  for (int i = 0; i < N; ++i) {
    poly = AddRealRoot(poly, roots[i]);
  }

  VectorXd real;
  bool success = FindPolynomialRootsJenkinsTraub(poly, &real, NULL);
  real = SortVector(real);
  EXPECT_EQ(success, true);
  EXPECT_EQ(real.size(), N);
  for (int i = 0; i < real.size(); i++) {
    EXPECT_NEAR(EvaluatePolynomial(poly, real[i]), 0, kEpsilonLoose);
  }
}

TEST(Polynomial, HardPolynomial1) {
  Eigen::VectorXd polynomial(11);
  Eigen::VectorXd roots_re(10);
  Eigen::VectorXd roots_im(10);

  polynomial(10) = -52412.8655144021;
  polynomial(9) = -28342.548095425875;
  polynomial(8) = 20409.84088622263;
  polynomial(7) = 25844.743360023815;
  polynomial(6) = 11474.831044766257;
  polynomial(5) = 1909.2968243041091;
  polynomial(4) = -692.3579951742573;
  polynomial(3) = -562.5089223272787;
  polynomial(2) = -105.89974320540716;
  polynomial(1) = 18.62488243410351;
  polynomial(0) = 5.576312106019016;

  EXPECT_TRUE(
      FindPolynomialRootsJenkinsTraub(polynomial, &roots_re, &roots_im));
}

TEST(Polynomial, HardPolynomial2) {
  Eigen::VectorXd polynomial(20);
  Eigen::VectorXd roots_re(19);
  Eigen::VectorXd roots_im(19);

  polynomial(19) = -3.3501738067312306e8;
  polynomial(18) = -6.884086124142883e8;
  polynomial(17) = 7.702813653628246e8;
  polynomial(16) = 8.451429594854779e8;
  polynomial(15) = -7.822417923012168e8;
  polynomial(14) = -2.0621500003041908e8;
  polynomial(13) = 2.71193932055516e8;
  polynomial(12) = 2191206.652049609;
  polynomial(11) = -4.3103846059516795e7;
  polynomial(10) = 3893518.9815099635;
  polynomial(9) = 4037788.101972703;
  polynomial(8) = -541891.2574823081;
  polynomial(7) = -260979.552665553;
  polynomial(6) = 38001.29427556511;
  polynomial(5) = 12074.712839195976;
  polynomial(4) = -1512.0183242937462;
  polynomial(3) = -388.5049059868163;
  polynomial(2) = 27.301047297669705;
  polynomial(1) = 6.8768381102442575;
  polynomial(0) = 0;

  EXPECT_TRUE(
      FindPolynomialRootsJenkinsTraub(polynomial, &roots_re, &roots_im));
}

// This test polynomial was provided by a user.
TEST(Polynomial, JenkinsTraub4Roots1) {
  const double roots[4] = {1.0843989379558703, 1.0844294564653463,
                           1.3072756126590779, 1.4643848994415114};
  RunPolynomialTestRealRoots(roots, true, false, kEpsilonLoose);
}

// This test polynomial was provided by a user.
TEST(Polynomial, JenkinsTraub4Roots2) {
  static const int N = 4;
  for (int j = 0; j < 10000; ++j) {
    VectorXd poly = ConstantPolynomial(1.23);
    VectorXd roots = VectorXd::Random(N);
    for (int i = 0; i < N; ++i) {
        roots(i) *= 0.5;
        roots(i) += 1.0;
    }

    roots = SortVector(roots);

    for (int i = 0; i < N; ++i) {
      poly = AddRealRoot(poly, roots[i]);
    }

    VectorXd real;
    const bool success = FindPolynomialRootsJenkinsTraub(poly, &real, NULL);
    EXPECT_EQ(success, true);
    real = SortVector(real);

    EXPECT_EQ(real.size(), N);
    for (int i = 0; i < real.size(); i++) {
      EXPECT_NEAR(EvaluatePolynomial(poly, real[i]), 0, kEpsilonLoose);
    }
  }
}


// Regression test for https://github.com/sweeneychris/RpolyPlusPlus/issues/5
// Torus ray-tracing quartics for which the solver previously returned all
// zeros (unguarded divisions poisoned the K-polynomial with NaN).
TEST(Polynomial, Issue5TorusQuartics) {
  const double quartics[12][5] = {
      {1, -198.995, 15006.6, -508127, 6.52025e+06},
      {1, -200.246, 15131.5, -511320, 6.52025e+06},
      {1, -202.193, 15327.2, -516281, 6.52025e+06},
      {1, -202.205, 15328.4, -516313, 6.52025e+06},
      {1, -202.215, 15329.4, -516336, 6.52025e+06},
      {1, -202.268, 15334.8, -516476, 6.52025e+06},
      {1, -202.289, 15336.8, -516525, 6.52025e+06},
      {1, -202.35,  15343.0, -516680, 6.52025e+06},
      {1, -202.389, 15347.0, -516781, 6.52025e+06},
      {1, -202.388, 15346.8, -516775, 6.52025e+06},
      {1, -202.427, 15350.8, -516876, 6.52025e+06},
      {1, -202.427, 15350.8, -516876, 6.52025e+06}};
  for (int k = 0; k < 12; k++) {
    VectorXd poly(5);
    for (int i = 0; i < 5; i++) poly(i) = quartics[k][i];
    VectorXd real_roots, complex_roots;
    ASSERT_TRUE(
        FindPolynomialRootsJenkinsTraub(poly, &real_roots, &complex_roots));
    // Every returned root must actually be a root: relative residual check.
    bool any_nonzero = false;
    for (int i = 0; i < real_roots.size(); i++) {
      const std::complex<double> z(real_roots(i), complex_roots(i));
      any_nonzero = any_nonzero || std::abs(z) > 0.0;
      std::complex<double> p = 0.0;
      double bound = 0.0;
      for (int j = 0; j < poly.size(); j++) {
        p = p * z + poly(j);
        bound = bound * std::abs(z) + std::abs(poly(j));
      }
      EXPECT_LT(std::abs(p), 1e-10 * bound);
    }
    // The all-zeros failure mode from the issue must not reappear.
    EXPECT_TRUE(any_nonzero);
  }
}

// Regression test for https://github.com/sweeneychris/RpolyPlusPlus/issues/8
// Degree-14 palindromic polynomial with tightly clustered roots on the unit
// circle. The solver previously returned incorrect roots; after the residual
// acceptance gate it must either solve correctly or fail loudly - and with
// stage-3 restarts it solves correctly.
TEST(Polynomial, Issue8ReciprocalPolynomial) {
  VectorXd poly(15);
  poly << 1, 0, -1, 0, 1, 1, 1, 1, 1, 1, 1, 0, -1, 0, 1;
  VectorXd real_roots, complex_roots;
  ASSERT_TRUE(
      FindPolynomialRootsJenkinsTraub(poly, &real_roots, &complex_roots));
  // Reference roots (companion-matrix eigenvalues, cross-checked against the
  // values reported in the issue). Palindromic: roots come in (z, 1/z) pairs.
  const double kTrueRoots[14][2] = {
      {1.205106, 0.579537},   {1.205106, -0.579537},
      {-0.546374, 0.837541},  {-0.546374, -0.837541},
      {-0.979328, 0.202277},  {-0.979328, -0.202277},
      {-0.850840, 0.525425},  {-0.850840, -0.525425},
      {0.533286, 0.845935},   {0.533286, -0.845935},
      {-0.035793, 0.999359},  {-0.035793, -0.999359},
      {0.673943, 0.324100},   {0.673943, -0.324100}};
  for (int t = 0; t < 14; t++) {
    const std::complex<double> truth(kTrueRoots[t][0], kTrueRoots[t][1]);
    double best = 1e300;
    for (int i = 0; i < real_roots.size(); i++) {
      best = std::min(best,
                      std::abs(std::complex<double>(real_roots(i),
                                                    complex_roots(i)) -
                               truth));
    }
    EXPECT_LT(best, 1e-5);  // truth table is 6-decimal precision
  }
}


namespace metamorphic_detail {

inline std::vector<std::complex<double>> SolveToVec(const VectorXd& poly,
                                                    bool* ok) {
  VectorXd re, im;
  *ok = FindPolynomialRootsJenkinsTraub(poly, &re, &im);
  std::vector<std::complex<double>> r;
  if (*ok) {
    for (int i = 0; i < re.size(); i++) r.emplace_back(re(i), im(i));
  }
  return r;
}

// Absolute condition of each root; identity checks are judged in units of
// eps * kappa (the best accuracy any solver can achieve for that root).
inline double RootCondition(const VectorXd& p, std::complex<double> z) {
  std::complex<double> v = 0, dv = 0;
  double bound = 0;
  const double az = std::abs(z);
  for (int i = 0; i < p.size(); i++) {
    dv = dv * z + v;
    v = v * z + p(i);
    bound = bound * az + std::abs(p(i));
  }
  const double d = std::abs(dv);
  return d > 0 ? bound / d : 1e300;
}

inline double SetDistInConditionUnits(
    const VectorXd& p_ref, const std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& b) {
  if (a.size() != b.size()) return 1e300;
  double worst = 0;
  std::vector<bool> used(b.size(), false);
  for (const auto& x : a) {
    double best = 1e300;
    int bi = -1;
    for (size_t j = 0; j < b.size(); j++) {
      if (used[j]) continue;
      const double d = std::abs(x - b[j]);
      if (d < best) { best = d; bi = static_cast<int>(j); }
    }
    if (bi >= 0) used[bi] = true;
    const double eps = std::numeric_limits<double>::epsilon();
    worst = std::max(worst,
                     best / (eps * std::max(1.0, RootCondition(p_ref, x))));
  }
  return worst;
}

// Universal per-root backward-error certificate: every returned root
// evaluates to rounding noise relative to sum_i |p_i||z|^i.
inline bool CertifiedBackwardStable(
    const VectorXd& p, const std::vector<std::complex<double>>& roots,
    double tol) {
  for (const auto& z : roots) {
    std::complex<double> v = 0;
    double bound = 0;
    const double az = std::abs(z);
    for (int i = 0; i < p.size(); i++) {
      v = v * z + p(i);
      bound = bound * az + std::abs(p(i));
    }
    if (std::abs(v) > tol * bound) return false;
  }
  return true;
}

}  // namespace metamorphic_detail

// Metamorphic invariants: solver output must respect the exact symmetries of
// the root-finding problem, judged against per-root conditioning. These need
// no ground-truth roots, so they scale to arbitrary random inputs.
TEST(Polynomial, MetamorphicInvariants) {
  using metamorphic_detail::SolveToVec;
  using metamorphic_detail::CertifiedBackwardStable;
  using metamorphic_detail::SetDistInConditionUnits;
  std::mt19937 rng(20260702);
  std::uniform_real_distribution<double> unif(-1, 1);
  std::uniform_int_distribution<int> coin(0, 1);
  for (int t = 0; t < 200; t++) {
    const int deg = 3 + (t % 10);
    std::vector<std::complex<double>> roots;
    while (static_cast<int>(roots.size()) < deg) {
      if (deg - static_cast<int>(roots.size()) >= 2 && coin(rng)) {
        const double a = unif(rng), b = std::abs(unif(rng)) + 0.05;
        roots.emplace_back(a, b);
        roots.emplace_back(a, -b);
      } else {
        double x = unif(rng);
        if (std::abs(x) < 0.05) x += 0.2;
        roots.emplace_back(x, 0);
      }
    }
    VectorXd poly = VectorXd::Zero(deg + 1);
    {
      std::vector<std::complex<double>> c = {1.0};
      for (const auto& r : roots) {
        std::vector<std::complex<double>> nc(c.size() + 1, 0.0);
        for (size_t i = 0; i < c.size(); i++) {
          nc[i] += c[i];
          nc[i + 1] -= r * c[i];
        }
        c = nc;
      }
      for (size_t i = 0; i < c.size(); i++) poly(i) = c[i].real();
    }
    bool ok = false;
    const auto base = SolveToVec(poly, &ok);
    ASSERT_TRUE(ok);

    // roots(p(-x)) == -roots(p)
    VectorXd pn = poly;
    for (int i = 0; i < poly.size(); i++) {
      if ((poly.size() - 1 - i) % 2 == 1) pn(i) = -pn(i);
    }
    bool okn = false;
    const auto rn = SolveToVec(pn, &okn);
    ASSERT_TRUE(okn);
    std::vector<std::complex<double>> neg;
    for (const auto& z : base) neg.push_back(-z);
    EXPECT_TRUE(SetDistInConditionUnits(pn, rn, neg) < 1e3 ||
                (CertifiedBackwardStable(pn, rn, 1e-11) &&
                 CertifiedBackwardStable(poly, base, 1e-11)))
        << "identity violated and solution not backward-stable";

    // roots(reverse(p)) == 1/roots(p)
    VectorXd pr = poly.reverse();
    bool okr = false;
    const auto rr = SolveToVec(pr, &okr);
    ASSERT_TRUE(okr);
    std::vector<std::complex<double>> inv;
    for (const auto& z : base) inv.push_back(1.0 / z);
    EXPECT_TRUE(SetDistInConditionUnits(pr, rr, inv) < 1e3 ||
                (CertifiedBackwardStable(pr, rr, 1e-11) &&
                 CertifiedBackwardStable(poly, base, 1e-11)))
        << "identity violated and solution not backward-stable";

    // roots(c * p) == roots(p)
    bool okc = false;
    const auto rc = SolveToVec((poly * 1e10).eval(), &okc);
    ASSERT_TRUE(okc);
    EXPECT_TRUE(SetDistInConditionUnits(poly, rc, base) < 1e3 ||
                (CertifiedBackwardStable(poly, rc, 1e-11) &&
                 CertifiedBackwardStable(poly, base, 1e-11)))
        << "identity violated and solution not backward-stable";
  }
}

// Mignotte polynomials x^n - 2(ax-1)^2: two roots near 1/a separated by
// ~a^{-n}; the canonical clustered-root stress test.
TEST(Polynomial, MignotteClusteredRoots) {
  for (const double a : {10.0, 100.0}) {
    const int n = 9;
    VectorXd poly = VectorXd::Zero(n + 1);
    poly(0) = 1;
    poly(n - 2) += -2 * a * a;
    poly(n - 1) += 4 * a;
    poly(n) += -2;
    VectorXd re, im;
    ASSERT_TRUE(FindPolynomialRootsJenkinsTraub(poly, &re, &im));
    for (int i = 0; i < re.size(); i++) {
      const std::complex<double> z(re(i), im(i));
      std::complex<double> v = 0;
      double bound = 0;
      for (int j = 0; j < poly.size(); j++) {
        v = v * z + poly(j);
        bound = bound * std::abs(z) + std::abs(poly(j));
      }
      EXPECT_LT(std::abs(v), 1e-7 * bound);
    }
  }
}

// Root-collision sweep ((x^2 - t) * (x^2 + 3x + 2)): a real root pair merges
// into a double root at t = 0 and splits into a complex pair for t < 0 --
// the tangency geometry underlying issue #5. The solver must remain correct
// through the transition, with accuracy degrading no worse than the
// square-root law of double-root conditioning.
TEST(Polynomial, TangencySweep) {
  for (const double sign : {1.0, -1.0}) {
    for (int e = 2; e <= 15; e++) {
      const double tt = sign * std::pow(10.0, -e);
      VectorXd poly(5);
      poly << 1, 3, 2 - tt, -3 * tt, -2 * tt;
      VectorXd re, im;
      ASSERT_TRUE(FindPolynomialRootsJenkinsTraub(poly, &re, &im));
      const std::complex<double> truth = std::sqrt(std::complex<double>(tt));
      double best = 1e300;
      for (int i = 0; i < re.size(); i++) {
        best = std::min(best,
                        std::abs(std::complex<double>(re(i), im(i)) - truth));
      }
      EXPECT_LT(best, 10 * (std::sqrt(std::abs(tt)) + 3e-8));
    }
  }
  VectorXd poly0(5);
  poly0 << 1, 3, 2, 0, 0;  // exact double zero root at the boundary
  VectorXd re, im;
  EXPECT_TRUE(FindPolynomialRootsJenkinsTraub(poly0, &re, &im));
}

}  // namespace rpoly_plus_plus
