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

// Tests for Newton root polishing at and near multiple roots.
//
// At a multiple root both p(x) and p'(x) are pure rounding noise, so the
// Newton step p/p' is arbitrary. The original polisher only stopped when p'
// was exactly 0.0 and could "polish" a root onto a *different* root, e.g.
// {1, 3, 7, 7} -> {1, 3, 3, 3}. That end-to-end failure depends on the last
// bits of the solver output (it reproduced only without FMA contraction), so
// end-to-end tests alone are a fragile detector. The tests below therefore
// exercise the polishing step directly over exhaustive families of start
// points, which deterministically catches the failure mode on any platform,
// and then check the full solver on the same families.
//
// What each group guards against:
//   *NeverMigrate*        - stopping too late (noise-driven steps).
//   *ScaleInvariant*      - a fixed absolute/relative threshold on p'.
//   *FullAccuracy*        - stopping too early (an over-conservative bound).
//   Solver*               - the end-to-end symptom, on real solver output.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "src/find_polynomial_roots_jenkins_traub.h"
#include "src/root_polishing.h"

namespace rpoly_plus_plus {
namespace {

using Eigen::VectorXd;
using internal::PolishRoot;
using Complex = std::complex<double>;

constexpr double kEps = std::numeric_limits<double>::epsilon();

// Integer roots in [-kRange, kRange]: coefficients are exact in double, and
// small integer roots are what made the noisy step p/p' hit other roots
// (ratios of few-ulp quantities are simple rationals).
constexpr int kRange = 10;

// Monic polynomial with the given roots, highest-degree coefficient first.
// Complex roots must come with their conjugates; the result is real.
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

VectorXd FromRealRoots(const std::vector<double>& roots) {
  return FromRoots(std::vector<Complex>(roots.begin(), roots.end()));
}

// Coefficients of s^n * p(x / s): the same polynomial with every root
// multiplied by s. Leading coefficient stays 1.
VectorXd ScaleRoots(const VectorXd& poly, double root_scale) {
  VectorXd scaled = poly;
  double f = 1.0;
  for (Eigen::Index i = 0; i < poly.size(); ++i) {
    scaled(i) *= f;
    f *= root_scale;
  }
  return scaled;
}

template <typename T>
T Horner(const VectorXd& poly, const T& x) {
  T v = 0.0;
  for (Eigen::Index i = 0; i < poly.size(); ++i) v = v * x + poly(i);
  return v;
}

// Best accuracy achievable for a *simple* root r in double precision:
// the evaluation noise eps * q(|r|) (q = absolute-value polynomial) divided
// by |p'(r)|, with a small safety factor, plus one ulp for representing r.
double AttainableAccuracy(const VectorXd& poly, const Complex& r) {
  double q = 0.0;
  Complex dp = 0.0, p = 0.0;
  for (Eigen::Index i = 0; i < poly.size(); ++i) {
    dp = dp * r + p;
    p = p * r + poly(i);
    q = q * std::abs(r) + std::abs(poly(i));
  }
  return 8.0 * poly.size() * kEps * q / std::abs(dp) +
         2.0 * kEps * std::abs(r);
}

// Visits every double within `ulps` units in the last place of `center`.
template <typename F>
void ForEachNearby(double center, int ulps, F&& f) {
  double x = center;
  for (int k = 0; k < ulps; ++k) x = std::nextafter(x, -HUGE_VAL);
  for (int k = 0; k <= 2 * ulps; ++k) {
    f(x);
    x = std::nextafter(x, HUGE_VAL);
  }
}

// Polishing a start point that is already within a few ulps of the multiple
// root `c` must leave it in c's cluster. Since all other roots are >= 1 away,
// the 1e-6 relative tolerance cleanly separates "stayed" from "migrated".
void ExpectRealStartsStayAtMultipleRoot(const VectorXd& poly, double c,
                                        int ulps) {
  const double tol = 1e-6 * std::max(1.0, std::abs(c));
  int failures = 0;
  ForEachNearby(c, ulps, [&](double x0) {
    const double x = PolishRoot(poly, x0);
    if (!(std::abs(x - c) <= tol) && ++failures <= 3) {
      ADD_FAILURE() << "multiple root " << c << " polished from "
                    << x0 << " to " << x << "; poly = "
                    << poly.transpose();
    }
    // Polishing never increases the residual, by construction.
    EXPECT_LE(std::abs(Horner(poly, x)), std::abs(Horner(poly, x0)));
  });
}

// ---------------------------------------------------------------------------
// Direct tests of the polishing step at multiple roots.
// ---------------------------------------------------------------------------

// The reported polynomials, with starts around every multiple root.
TEST(RootPolishing, ReportedPolynomialsNeverMigrate) {
  ExpectRealStartsStayAtMultipleRoot(FromRealRoots({1, 3, 7, 7}), 7, 256);
  const VectorXd p2 = FromRealRoots({-8, -8, -4, -4, 1, 4, 10});
  ExpectRealStartsStayAtMultipleRoot(p2, -8, 256);
  ExpectRealStartsStayAtMultipleRoot(p2, -4, 256);
}

// Exhaustive families of real multiple roots: double, triple, and two double
// roots, alongside simple roots, over all small-integer root placements.
TEST(RootPolishing, RealMultipleRootFamiliesNeverMigrate) {
  constexpr int kUlps = 32;
  for (int a = -kRange; a <= kRange; ++a) {
    for (int c = -kRange; c <= kRange; ++c) {
      if (c == a) continue;
      // (x-a)(x-c)^3
      ExpectRealStartsStayAtMultipleRoot(FromRealRoots({1.0 * a, 1.0 * c,
                                                        1.0 * c, 1.0 * c}),
                                         c, kUlps);
      for (int b = a + 1; b <= kRange; ++b) {
        if (b == c) continue;
        // (x-a)(x-b)(x-c)^2 -- the family of the first report.
        ExpectRealStartsStayAtMultipleRoot(
            FromRealRoots({1.0 * a, 1.0 * b, 1.0 * c, 1.0 * c}), c, kUlps);
        // (x-a)^2(x-b)(x-c)^2 -- two double roots, as in the second report.
        if (a < c) {
          const VectorXd p = FromRealRoots(
              {1.0 * a, 1.0 * a, 1.0 * b, 1.0 * c, 1.0 * c});
          ExpectRealStartsStayAtMultipleRoot(p, a, kUlps);
          ExpectRealStartsStayAtMultipleRoot(p, c, kUlps);
        }
      }
    }
  }
}

// Complex multiple roots (repeated conjugate pairs) go through the complex
// polishing path. Starts are a grid of ulp offsets in both components.
TEST(RootPolishing, ComplexMultipleRootFamiliesNeverMigrate) {
  constexpr int kUlps = 8;
  for (int re = -5; re <= 5; ++re) {
    for (int im = 1; im <= 5; ++im) {
      const Complex z(re, im);
      for (int a = -kRange; a <= kRange; ++a) {
        for (int b = a; b <= kRange; b += 3) {
          // (x-z)^2 (x-conj z)^2 (x-a)(x-b)
          const VectorXd poly =
              FromRoots({z, std::conj(z), z, std::conj(z), Complex(a),
                         Complex(b)});
          int failures = 0;
          ForEachNearby(z.real(), kUlps, [&](double x) {
            ForEachNearby(z.imag(), kUlps, [&](double y) {
              const Complex z0(x, y);
              const Complex z1 = PolishRoot(poly, z0);
              if (!(std::abs(z1 - z) <= 1e-6 * std::abs(z)) &&
                  ++failures <= 3) {
                ADD_FAILURE() << "complex double root " << z
                              << " polished from " << z0 << " to " << z1
                              << "; poly = " << poly.transpose();
              }
            });
          });
        }
      }
    }
  }
}

// The noise threshold must scale with the polynomial. A fixed cutoff on p'
// (e.g. |p'| <= 1e-14 or 1e-10) is right for at most one scale: too small for
// large roots (migration returns) and too large for small roots (simple roots
// stop being polished; see SimpleRootsPolishToFullAccuracyAtAnyScale).
TEST(RootPolishing, MultipleRootsNeverMigrateAtAnyScale) {
  // Non-powers of two, so scaling is not exact in binary.
  const double kScales[] = {1e-6, 3.7e-3, 0.3, 7.3, 1.1e3, 1e6};
  for (const double s : kScales) {
    for (int a = -kRange; a <= kRange; ++a) {
      for (int b = a + 1; b <= kRange; ++b) {
        for (int c = -kRange; c <= kRange; ++c) {
          if (c == a || c == b) continue;
          // Roots {a, b, c, c} * s.
          const VectorXd poly =
              ScaleRoots(FromRealRoots({1.0 * a, 1.0 * b, 1.0 * c, 1.0 * c}),
                         s);
          const double cs = c * s;
          const double tol = 1e-6 * std::max(1.0, std::abs(1.0 * c)) * s;
          int failures = 0;
          ForEachNearby(cs, 8, [&](double x0) {
            const double x = PolishRoot(poly, x0);
            if (!(std::abs(x - cs) <= tol) && ++failures <= 3) {
              ADD_FAILURE() << "scale " << s << ": multiple root " << cs
                            << " polished from " << x0 << " to " << x;
            }
          });
        }
      }
    }
  }
}

// The same must hold when only the coefficients are scaled (p and c*p have
// the same roots), down to the edges of the double range.
TEST(RootPolishing, MultipleRootsNeverMigrateUnderCoefficientScaling) {
  const double kCoeffScales[] = {1e-300, 1e-150, 1e-20, 1e20, 1e150, 1e300};
  const VectorXd base = FromRealRoots({1, 3, 7, 7});
  for (const double k : kCoeffScales) {
    ExpectRealStartsStayAtMultipleRoot(VectorXd(base * k), 7, 256);
  }
}

// ---------------------------------------------------------------------------
// The other side: polishing must not stop too early. Simple roots, including
// tightly clustered ones whose p' is small but meaningful, must still be
// polished to the accuracy that double precision allows.
// ---------------------------------------------------------------------------

// Starts at distances from `max_start_error` down to ~1e-7 of it, spanning
// what the Jenkins-Traub iteration hands over. Polishing only refines: the
// start must be well inside Newton's basin (much closer than the nearest
// other root), where two steps converge quadratically.
void ExpectPolishesToFullAccuracy(const VectorXd& poly, const Complex& r,
                                  const double max_start_error) {
  const double tol = AttainableAccuracy(poly, r);
  for (const double f : {1.0, -1e-2, 1e-5, -1e-7}) {
    const double offset = f * max_start_error;
    if (r.imag() == 0.0) {
      const double x0 = r.real() + offset;
      const double x = PolishRoot(poly, x0);
      EXPECT_LE(std::abs(x - r.real()), tol)
          << "simple root " << r.real() << " from " << x0
          << "; poly = " << poly.transpose();
    } else {
      const Complex z0 = r + Complex(offset, -offset) / std::sqrt(2.0);
      const Complex z = PolishRoot(poly, z0);
      EXPECT_LE(std::abs(z - r), tol)
          << "simple root " << r << " from " << z0
          << "; poly = " << poly.transpose();
    }
  }
}

TEST(RootPolishing, SimpleRootsPolishToFullAccuracyAtAnyScale) {
  const double kScales[] = {1e-6, 3.7e-3, 1.0, 7.3, 1.1e3, 1e6};
  for (const double s : kScales) {
    for (int a = -kRange; a <= kRange; a += 2) {
      for (int b = a + 1; b <= kRange; b += 3) {
        for (int c = b + 1; c <= kRange; c += 2) {
          const VectorXd poly = ScaleRoots(
              FromRealRoots({1.0 * a, 1.0 * b, 1.0 * c, 1.0 * c + 0.5}),
              s);
          for (const double r : {a, b, c}) {
            // Exact zero roots are deflated before polishing, never polished.
            if (r == 0) continue;
            ExpectPolishesToFullAccuracy(poly, r * s,
                                         1e-6 * std::max(std::abs(r), 1.0) * s);
          }
        }
      }
    }
    // A complex pair among real roots.
    const Complex z(2, 3);
    const VectorXd poly =
        ScaleRoots(FromRoots({z, std::conj(z), -4.0, 1.0, 6.0}), s);
    ExpectPolishesToFullAccuracy(poly, z * s, 1e-6 * std::abs(z * s));
    ExpectPolishesToFullAccuracy(poly, -4.0 * s, 4e-6 * s);
  }
}

// Nearly-multiple roots: p' at each root is small (~delta) but well above its
// noise, so the polisher must still refine them -- this is what separates a
// noise bound from a crude "p' is small" cutoff.
TEST(RootPolishing, NearlyMultipleRootsPolishToFullAccuracy) {
  for (const double delta : {1e-2, 1e-4, 1e-5}) {
    for (const double c : {-8.0, 0.5, 7.0}) {
      const VectorXd poly = FromRealRoots({1.0, 3.0, c, c + delta});
      ExpectPolishesToFullAccuracy(poly, c, 1e-3 * delta);
    }
    const Complex z(1, 2), w = z + Complex(delta, 0);
    ExpectPolishesToFullAccuracy(
        FromRoots({z, std::conj(z), w, std::conj(w), 5.0}), z, 1e-3 * delta);
  }
}

// ---------------------------------------------------------------------------
// End-to-end: the full solver on the same kinds of polynomials.
// ---------------------------------------------------------------------------

// Each true root must be matched by exactly as many computed roots as its
// multiplicity. Multiple roots are only determined to ~eps^(1/m), so the match
// radius is loose; distinct roots are >= 1 apart, so migration still fails.
void ExpectSolverFindsRootsWithMultiplicity(const std::vector<Complex>& roots) {
  const VectorXd poly = FromRoots(roots);
  VectorXd re, im;
  ASSERT_TRUE(FindPolynomialRootsJenkinsTraub(poly, &re, &im))
      << "poly = " << poly.transpose();
  ASSERT_EQ(re.size(), static_cast<Eigen::Index>(roots.size()));
  for (const Complex& r : roots) {
    const int multiplicity =
        static_cast<int>(std::count(roots.begin(), roots.end(), r));
    const double tol = 1e-3 * std::max(1.0, std::abs(r));
    int found = 0;
    for (Eigen::Index i = 0; i < re.size(); ++i) {
      if (std::abs(Complex(re(i), im(i)) - r) <= tol) ++found;
    }
    EXPECT_EQ(found, multiplicity)
        << "root " << r << " in poly " << poly.transpose()
        << "\n  computed re: " << re.transpose()
        << "\n  computed im: " << im.transpose();
  }
}

// The two user reports. Built without FMA contraction (GCC on x86-64, or the
// *_nofma_test target) these returned {1, 3, 3, 3} and
// {-8, -8, -8, -8, 1, 4, 10}.
TEST(RootPolishing, SolverReportedPolynomials) {
  ExpectSolverFindsRootsWithMultiplicity({1.0, 3.0, 7.0, 7.0});
  ExpectSolverFindsRootsWithMultiplicity(
      {-8.0, -8.0, -4.0, -4.0, 1.0, 4.0, 10.0});
}

TEST(RootPolishing, SolverRealMultipleRootFamilies) {
  for (int a = -kRange; a <= kRange; ++a) {
    for (int b = a + 1; b <= kRange; ++b) {
      for (int c = -kRange; c <= kRange; ++c) {
        if (c == a || c == b) continue;
        const Complex A(a), B(b), C(c);
        ExpectSolverFindsRootsWithMultiplicity({A, B, C, C});
        ExpectSolverFindsRootsWithMultiplicity({A, A, B, B, C});
        ExpectSolverFindsRootsWithMultiplicity({A, B, C, C, C});
        ExpectSolverFindsRootsWithMultiplicity({A, A, B, B, C, C});
      }
    }
  }
}

TEST(RootPolishing, SolverComplexMultipleRootFamilies) {
  for (int re = -5; re <= 5; ++re) {
    for (int im = 1; im <= 5; ++im) {
      const Complex z(re, im), zc = std::conj(z);
      for (int a = -kRange; a <= kRange; ++a) {
        ExpectSolverFindsRootsWithMultiplicity({z, zc, Complex(a), Complex(a)});
        for (int b = a + 1; b <= kRange; ++b) {
          ExpectSolverFindsRootsWithMultiplicity(
              {z, zc, z, zc, Complex(a), Complex(b)});
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Real double roots must come back real, not as a near-real complex pair.
// ---------------------------------------------------------------------------

// Rounding noise of size delta splits a double root a into
// a +/- sqrt(-delta / k), k = P''(a)/2. That is a complex pair whenever delta
// has the "wrong" sign, which happened for ~10% of the family below (e.g.
// (x+10)^2 (x+9)(x+7) gave -10 +/- 6.3e-7i). Callers that pick out real roots
// with im == 0 then silently lost both of them.
void ExpectAllRootsReal(const std::vector<double>& roots, double scale) {
  const VectorXd poly = ScaleRoots(FromRealRoots(roots), scale);
  VectorXd re, im;
  ASSERT_TRUE(FindPolynomialRootsJenkinsTraub(poly, &re, &im))
      << "poly = " << poly.transpose();
  for (Eigen::Index i = 0; i < im.size(); ++i) {
    if (im(i) != 0.0) {
      ADD_FAILURE() << "real root returned as complex " << re(i) << " + "
                    << im(i) << "i; poly = " << poly.transpose();
      return;
    }
  }
}

TEST(RootPolishing, SolverReturnsRealDoubleRootsAsReal) {
  for (const double scale : {1.0, 1e-6, 1e6}) {
    for (int a = -kRange; a <= kRange; ++a) {
      for (int b = -kRange; b <= kRange; ++b) {
        for (int c = b + 1; c <= kRange; ++c) {
          if (a == b || a == c) continue;
          ExpectAllRootsReal({1.0 * a, 1.0 * a, 1.0 * b, 1.0 * c}, scale);
          if (b != 0 && c != 0) {
            ExpectAllRootsReal({1.0 * a, 1.0 * a, 1.0 * b, 1.0 * b, 1.0 * c},
                               scale);
          }
        }
      }
    }
  }
}

// The double-root decision must not swallow genuine complex pairs. For the
// exact pair a +/- bi of (x^2 - 2ax + a^2 + b^2)(x - c)(x - d), the residual
// at the real critical point is b^2 * |(a-c)(a-d)|, so the pair is only
// indistinguishable from a double root when b is below the noise-cluster
// scale sqrt(eps * q(|a|) / |(a-c)(a-d)|). Above it (with margin) the pair
// must be kept. Below it, a split double root must be recognized.
TEST(RootPolishing, SplitDoubleRootDecisionRespectsNoiseScale) {
  using internal::IsSplitRealDoubleRoot;
  int kept = 0, merged = 0;
  for (int a = -kRange; a <= kRange; ++a) {
    if (a == 0) continue;
    for (int c = -kRange; c <= kRange; ++c) {
      for (int d = c + 1; d <= kRange; ++d) {
        if (a == c || a == d) continue;
        const VectorXd dbl = FromRealRoots({1.0 * a, 1.0 * a, 1.0 * c, 1.0 * d});
        double q = 0.0;
        for (Eigen::Index i = 0; i < dbl.size(); ++i) {
          q = q * std::abs(a) + std::abs(dbl(i));
        }
        const double cluster =
            std::sqrt(kEps * q / std::abs((a - c) * (a - d)));
        double x = 0.0;
        // Split double root, as the solver produces it (measured splits are
        // up to ~6x the cluster scale), with a slightly-off real part.
        const Complex split(a + 1e-11 * a, 4.0 * cluster);
        if (IsSplitRealDoubleRoot(dbl, split, &x)) {
          ++merged;
          EXPECT_NEAR(x, a, 1e-12 * std::abs(a));
        } else {
          ADD_FAILURE() << "split double root " << split << " not merged; "
                        << "poly = " << dbl.transpose();
        }
        // Genuine pair well outside the noise cluster.
        for (const double b : {100.0 * cluster, 1e-4 * std::abs(a)}) {
          const VectorXd pair = FromRoots(
              {Complex(a, b), Complex(a, -b), Complex(c), Complex(d)});
          if (IsSplitRealDoubleRoot(pair, Complex(a, b), &x)) {
            ADD_FAILURE() << "genuine pair " << a << " +/- " << b
                          << "i merged into a double root; poly = "
                          << pair.transpose();
          } else {
            ++kept;
          }
        }
      }
    }
  }
  EXPECT_GT(merged, 0);
  EXPECT_GT(kept, 0);
}

}  // namespace
}  // namespace rpoly_plus_plus
