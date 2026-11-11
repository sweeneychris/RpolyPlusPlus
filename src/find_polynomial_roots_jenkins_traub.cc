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

#include "src/find_polynomial_roots_jenkins_traub.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "src/polynomial.h"
#include "src/root_polishing.h"

namespace rpoly_plus_plus {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Vector3cd;
using Eigen::VectorXcd;

namespace {

// Maximum polynomial size (degree + 1) that is solved entirely on the stack.
// Above this, the solver falls back to heap-backed vectors.
constexpr int kStackCoeffs = 33;

// Dynamic-size, fixed-capacity vector: resize() only writes a size integer,
// storage is inline (stack when the owner is on the stack), and the full
// Eigen expression API is preserved.
using StackVec =
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, kStackCoeffs, 1>;

// Local polynomial evaluation that accepts any Eigen vector type without
// forcing a conversion/copy to VectorXd.
template <typename Derived, typename T>
inline T EvalPoly(const Eigen::MatrixBase<Derived>& polynomial, const T& x) {
  T v = 0.0;
  for (Eigen::Index i = 0; i < polynomial.size(); ++i) {
    v = v * x + polynomial(i);
  }
  return v;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// Machine precision constants.
constexpr double kAbsoluteTolerance = 1e-14;
constexpr double kRelativeTolerance = 1e-10;


// Normalize a polynomial so its first *non-negligible* coefficient is 1.
// Dividing by an (effectively) zero leading coefficient poisons the
// K-polynomial with inf/NaN, which is what happens for e.g. x^4 - 1.
template <typename Vec>
inline void SafeMonicNormalize(Vec* poly) {
  const double max_abs = poly->cwiseAbs().maxCoeff();
  if (max_abs == 0.0 || !std::isfinite(max_abs)) {
    return;
  }
  // Normalize by the first non-negligible coefficient. Only *structural* zeros
  // (exactly 0.0) are skipped; small-but-nonzero leading coefficients are
  // meaningful for polynomials with wide dynamic range (e.g. Wilkinson-type).
  int lead = 0;
  while (lead < poly->size() - 1 && (*poly)(lead) == 0.0) {
    ++lead;
  }
  // Reciprocal-multiply (n multiplies + one divide) instead of n divides:
  // profiling showed the per-element divisions dominated this routine, which
  // runs every shift iteration. The one way this can overflow is a tiny
  // leading coefficient scaling a large element past DBL_MAX; guard against
  // that by checking the largest resulting magnitude *before* mutating, and
  // fall back to scaling by the max magnitude (which keeps every entry <= 1 and
  // finite) -- the same safety the divide-then-allFinite form provided.
  const double inv = 1.0 / (*poly)(lead);
  if (max_abs * std::abs(inv) <= std::numeric_limits<double>::max()) {
    *poly *= inv;
  } else {
    *poly /= max_abs;
  }
}

enum ConvergenceType{
  NO_CONVERGENCE = 0,
  LINEAR_CONVERGENCE = 1,
  QUADRATIC_CONVERGENCE = 2
};

// Solves for the root of the equation ax + b = 0.
double FindLinearPolynomialRoots(const double a, const double b) {
  return -b / a;
}

// Stable quadratic roots according to BKP Horn.
// http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
void FindQuadraticPolynomialRoots(const double a,
                                  const double b,
                                  const double c,
                                  std::complex<double>* roots) {
  double D = b * b - 4 * a * c;
  double sqrt_D = std::sqrt(std::abs(D));
  if (!std::isfinite(D)) {
    // b*b or 4ac overflowed (D is inf, or NaN if both did) although the
    // roots may be representable (e.g. x^2 + 1e200 x + 1). Rare, so the
    // common path above stays branch-light: recompute with the terms scaled
    // by s ~ max(|b|, sqrt|4ac|).
    const double s = std::max(std::abs(b),
                              2.0 * std::sqrt(std::abs(a)) *
                                  std::sqrt(std::abs(c)));
    const double bs = b / s;
    D = bs * bs - 4.0 * (a / s) * (c / s);
    sqrt_D = s * std::sqrt(std::abs(D));
  }

  // Real roots.
  if (D >= 0) {
    if (b >= 0) {
      roots[0] = std::complex<double>((-b - sqrt_D) / (2.0 * a), 0);
      roots[1] = std::complex<double>((2.0 * c) / (-b - sqrt_D), 0);
    } else {
      roots[0] = std::complex<double>((2.0 * c) / (-b + sqrt_D), 0);
      roots[1] = std::complex<double>((-b + sqrt_D) / (2.0 * a), 0);
    }
    return;
  }

  // Use the normal quadratic formula for the complex case.
  roots[0] = std::complex<double>(-b / (2.0 * a), sqrt_D / (2.0 * a));
  roots[1] = std::complex<double>(-b / (2.0 * a), -sqrt_D / (2.0 * a));
}

// Perform division by a linear term of the form (z - x) and evaluate P at x.
template <typename Vec>
void SyntheticDivisionAndEvaluate(const Vec& polynomial,
                                  const double x,
                                  Vec* quotient,
                                  double* eval) {
  quotient->resize(polynomial.size() - 1);
  (*quotient)(0) = polynomial(0);
  for (Eigen::Index i = 1; i < polynomial.size() - 1; i++) {
    (*quotient)(i) = polynomial(i) + (*quotient)(i - 1) * x;
  }
  *eval = polynomial(polynomial.size() - 1) +
          (*quotient)(quotient->size() - 1) * x;
}

// Perform division of a polynomial by a quadratic factor. The quadratic divisor
// should have leading 1s.
template <typename Vec>
void QuadraticSyntheticDivision(const Vec& polynomial,
                                const Vector3d& quadratic_divisor,
                                Vec* quotient,
                                Eigen::Vector2d* remainder) {
  quotient->resize(polynomial.size() - 2);

  (*quotient)(0) = polynomial(0);

  // If the quotient is a constant then polynomial is degree 2 and the math is
  // simple.
  if (quotient->size() == 1) {
    *remainder = polynomial.template tail<2>() -
                 polynomial(0) * quadratic_divisor.tail<2>();
    return;
  }

  (*quotient)(1) = polynomial(1) - polynomial(0) * quadratic_divisor(1);
  for (Eigen::Index i = 2; i < polynomial.size() - 2; i++) {
    (*quotient)(i) = polynomial(i) - (*quotient)(i - 2) * quadratic_divisor(2) -
                     (*quotient)(i - 1) * quadratic_divisor(1);
  }

  const Eigen::Index qn = quotient->size();
  const Eigen::Index pn = polynomial.size();
  (*remainder)(0) = polynomial(pn - 2) -
                    quadratic_divisor(1) * (*quotient)(qn - 1) -
                    quadratic_divisor(2) * (*quotient)(qn - 2);
  (*remainder)(1) =
      polynomial(pn - 1) - quadratic_divisor(2) * (*quotient)(qn - 1);
}

// Determines whether the iteration has converged by examining the three most
// recent values for convergence.
template<typename T>
bool HasConverged(const T& sequence) {
  // Compare squared magnitudes so complex sequences avoid std::abs's hypot():
  // |d| < |a|/2  <=>  |d|^2 < |a|^2/4, both sides nonnegative. std::norm is
  // |x|^2 for complex and x*x for real, so this serves the real (sigma) and
  // complex (t) convergence sequences alike.
  const bool convergence_condition_1 =
      std::norm(sequence(1) - sequence(0)) < std::norm(sequence(0)) / 4.0;
  const bool convergence_condition_2 =
      std::norm(sequence(2) - sequence(1)) < std::norm(sequence(1)) / 4.0;

  // If the sequence has converged then return true.
  return convergence_condition_1 && convergence_condition_2;
}

// Determines if the root has converged by measuring the relative and absolute
// change in the root value. This stopping criterion is a simple measurement
// that proves to work well. It is referred to as "Ward's method" in the
// following reference:
//
// Nikolajsen, Jorgen L. "New stopping criteria for iterative root finding."
// Royal Society open science (2014)
// Fixed-capacity history of the three most recent root estimates. Replaces
// std::vector push_back/erase, which was the last source of heap traffic.
template <typename T>
struct RootHistory {
  T v[3];
  int n = 0;
  inline void push(const T& x) {
    if (n < 3) {
      v[n++] = x;
    } else {
      v[0] = v[1];
      v[1] = v[2];
      v[2] = x;
    }
  }
};

template <typename T>
bool HasRootConverged(const RootHistory<T>& roots) {
  constexpr double kRootMagnitudeTolerance = 1e-8;
  if (roots.n != 3) {
    return false;
  }

  // Squared magnitudes avoid hypot() on complex root histories; each
  // comparison is between nonnegative quantities, so squaring is exact.
  const double e_i2 = std::norm(roots.v[2] - roots.v[1]);
  const double e_i_minus_1_2 = std::norm(roots.v[1] - roots.v[0]);
  const double mag_root2 = std::norm(roots.v[1]);
  if (e_i2 <= e_i_minus_1_2) {
    if (mag_root2 < kRootMagnitudeTolerance * kRootMagnitudeTolerance) {
      return e_i2 < kAbsoluteTolerance * kAbsoluteTolerance;
    } else {
      return e_i2 <= kRelativeTolerance * kRelativeTolerance * mag_root2;
    }
  }

  return false;
}


// Backward-error acceptance gate. Ward's criterion only tests iterate
// stagnation; a stalled iteration far from any root can satisfy it. A root is
// only accepted if |P(s)| is consistent with rounding error in evaluating P
// at s, i.e. |P(s)| <= kGate * eps * sum_i |p_i| |s|^{n-i}.
template <typename Vec>
inline bool ResidualAcceptable(const Vec& polynomial,
                               const std::complex<double>& root) {
  constexpr double kGate = 1e3 * std::numeric_limits<double>::epsilon();
  std::complex<double> p = 0.0;
  double bound = 0.0;
  const double az = std::abs(root);
  for (Eigen::Index i = 0; i < polynomial.size(); ++i) {
    p = p * root + polynomial(i);
    bound = bound * az + std::abs(polynomial(i));
  }
  return std::abs(p) <= kGate * bound;
}

// Real-root overload: pure real Horner, no complex arithmetic and no hypot.
// For a real root of a real polynomial this is bit-identical to the complex
// version evaluated at (root, 0), just cheaper.
template <typename Vec>
inline bool ResidualAcceptable(const Vec& polynomial, double root) {
  constexpr double kGate = 1e3 * std::numeric_limits<double>::epsilon();
  double p = 0.0;
  double bound = 0.0;
  const double az = std::abs(root);
  for (Eigen::Index i = 0; i < polynomial.size(); ++i) {
    p = p * root + polynomial(i);
    bound = bound * az + std::abs(polynomial(i));
  }
  return std::abs(p) <= kGate * bound;
}

// Implementation closely follows the three-stage algorithm for finding roots of
// polynomials with real coefficients as outlined in: "A Three-Stage Algorithm
// for Real Polynomaials Using Quadratic Iteration" by Jenkins and Traub, SIAM
// 1970. Please note that this variant is different than the complex-coefficient
// version, and is estimated to be up to 4 times faster.
template <typename Vec>
class JenkinsTraubSolver {
 public:
  JenkinsTraubSolver(const Vec& coeffs,
                     VectorXd* real_roots,
                     VectorXd* complex_roots)
      : polynomial_(coeffs),
        real_roots_(real_roots),
        complex_roots_(complex_roots) {}

  // Extracts the roots using the Jenkins Traub method.
  bool ExtractRoots();

 private:
  // Removes any zero roots and divides polynomial by z.
  void RemoveZeroRoots();

  // Computes the magnitude of the roots to provide and initial search radius
  // for the iterative solver.
  double ComputeRootRadius();

  // Computes the zero-shift applied to the K-Polynomial.
  void ComputeZeroShiftKPolynomial();

  // Stage 1 of the Jenkins-Traub method. This stage is not technically
  // necessary, but helps separate roots that are close to zero.
  void ApplyZeroShiftToKPolynomial(const int num_iterations);

  // Computes and returns the update of sigma(z) based on the current
  // K-polynomial.
  //
  // NOTE: This function is used by the fixed shift iterations (which hold sigma
  // constant) so sigma is *not* modified internally by this function. If you
  // want to change sigma, simply call
  //    sigma = ComputeNextSigma();
  Vector3d ComputeNextSigma();

  // Updates the K-polynomial based on the current value of sigma for the fixed
  // or variable shift stage.
  void UpdateKPolynomialWithQuadraticShift(
      const Vec& polynomial_quotient,
      const Vec& k_polynomial_quotient);

  // Apply fixed-shift iterations to the K-polynomial to separate the
  // roots. Based on the convergence of the K-polynomial, we apply a
  // variable-shift linear or quadratic iteration to determine a real root or
  // complex conjugate pair of roots respectively.
  ConvergenceType ApplyFixedShiftToKPolynomial(const std::complex<double>& root,
                                               const int max_iterations);

  // Applies one of the variable shifts to the K-Polynomial. Returns true upon
  // successful convergence to a good root, and false otherwise.
  bool ApplyVariableShiftToKPolynomial(
      const ConvergenceType& fixed_shift_convergence,
      const std::complex<double>& root);

  // Applies a quadratic shift to the K-polynomial to determine a pair of roots
  // that are complex conjugates. Return true if a root was successfully found.
  bool ApplyQuadraticShiftToKPolynomial(const std::complex<double>& root,
                                        const int max_iterations);

  // Applies a linear shift to the K-polynomial to determine a single real root.
  // Return true if a root was successfully found.
  bool ApplyLinearShiftToKPolynomial(const std::complex<double>& root,
                                     const int max_iterations);

  // Adds the root to the output variables.
  void AddRootToOutput(const double real, const double imag);

  // Solves polynomials of degree <= 2.
  bool SolveClosedFormPolynomial();

  // Newton-polishes all output roots against the original (undeflated)
  // polynomial to remove deflation error accumulation.
  void PolishRoots();

  // Helper variables to manage the polynomials as they are being manipulated
  // and deflated.
  Vec polynomial_;
  Vec k_polynomial_;
  // The normalized input polynomial before any deflation, for root polishing.
  Vec original_polynomial_;
  // Preallocated scratch buffers reused across all iterations.
  Vec polynomial_quotient_, k_polynomial_quotient_;
  Vec deflated_polynomial_, deflated_k_polynomial_;
  Eigen::Vector2d polynomial_remainder_, k_polynomial_remainder_;
  // Sigma is the quadratic factor the divides the K-polynomial.
  Vector3d sigma_;

  // Let us define a, b, c, and d such that:
  //   P(z) = Q_P * sigma(z) + b * (z + u) + a
  //   K(z) = Q_K * sigma(z) + d * (z + u ) + c
  //
  // where Q_P and Q_K are the quotients from polynomial division of
  // sigma(z). Note that this means for a given a root s of sigma:
  //
  //   P(s)      = a - b * s_conj
  //   P(s_conj) = a - b * s
  //   K(s)      = c - d * s_conj
  //   K(s_conj) = c - d * s
  double a_, b_, c_, d_;

  // Output reference variables.
  VectorXd* real_roots_;
  VectorXd* complex_roots_;
  int num_solved_roots_ = 0;

  // Exponent k of the exact power-of-two variable scaling z = 2^k * w applied
  // to keep intermediate quantities in floating-point range for polynomials
  // with extreme root magnitudes. Roots are scaled back by 2^k on output.
  int scale_exponent_ = 0;

  // Keeps track of whether the linear and quadratic shifts have been attempted
  // yet so that we do not attempt the same shift twice.
  bool attempted_linear_shift_;
  bool attempted_quadratic_shift_;

  // Both shift budgets exhausted without Ward convergence: certify the best
  // tracked candidate with the exact residual gate.
  bool TryResidualFallback();

  // Last-resort rescue at restart exhaustion: gate-protected Newton seeded
  // from the trailing quadratic (captures clusters of roots much smaller
  // than the rest, e.g. near-tangent intersections at scale sqrt(t)).
  bool TrySmallRootRescue();
  bool CertifyAndDeflateReal(double candidate_root);
  bool CertifyAndDeflatePair(const double* candidate_sigma);
  // Tracking uses only O(1) proxies already computed by each iteration; the
  // single exact O(n) residual certification happens at fallback time.
  double best_real_metric_ = std::numeric_limits<double>::max();
  double best_real_root_ = 0.0;
  double best_pair_metric_ = std::numeric_limits<double>::max();
  double best_pair_sigma_[3] = {0.0, 0.0, 0.0};

  // Number of zero-shift iterations to perform.
  static constexpr int kNumZeroShiftIterations = 5;

  // The number of fixed shift iterations is computed as
  //   # roots found * this multiplier.
  static constexpr int kFixedShiftIterationMultiplier = 20;

  // If the fixed shift iterations fail to converge, we restart this many times
  // before considering the solve attempt as a failure.
  static constexpr int kMaxFixedShiftRestarts = 20;

  // The maximum number of linear shift iterations to perform before considering
  // the shift as a failure.
  static constexpr int kMaxLinearShiftIterations = 60;

  // The maximum number of quadratic shift iterations to perform before
  // considering the shift as a failure.
  static constexpr int kMaxQuadraticShiftIterations = 60;

  // When quadratic shift iterations are stalling, we attempt a few fixed shift
  // iterations to help convergence.
  static constexpr int kInnerFixedShiftIterations = 5;

  // During quadratic iterations, the real values of the root pairs should be
  // nearly equal since the root pairs are complex conjugates. This tolerance
  // measures how much the real values may diverge before consider the quadratic
  // shift to be failed.
  static constexpr double kRootPairTolerance = 0.01;
};

template <typename Vec>
bool JenkinsTraubSolver<Vec>::ExtractRoots() {
  if (polynomial_.size() == 0) {
    std::cout << "Invalid polynomial of size 0 passed to "
                 "FindPolynomialRootsJenkinsTraub" << std::endl;
    return false;
  }

  // Remove any leading zeros of the polynomial (in place).
  {
    Eigen::Index lead = 0;
    while (lead < polynomial_.size() - 1 && polynomial_(lead) == 0) ++lead;
    if (lead > 0) {
      const Eigen::Index new_size = polynomial_.size() - lead;
      for (Eigen::Index i = 0; i < new_size; ++i) {
        polynomial_(i) = polynomial_(i + lead);
      }
      polynomial_.conservativeResize(new_size);
    }
  }
  // Normalize the polynomial.
  polynomial_ /= polynomial_(0);
  const int degree = polynomial_.size() - 1;

  // Allocate the output roots.
  if (real_roots_ != NULL) {
    real_roots_->setZero(degree);
  }
  if (complex_roots_ != NULL) {
    complex_roots_->setZero(degree);
  }

  // Remove any zero roots.
  RemoveZeroRoots();

  // Exact power-of-two variable scaling z = 2^k * w for extreme root
  // magnitudes, so intermediate quantities (|root|^2 in sigma, 4th-power terms
  // in ComputeNextSigma) stay in floating-point range. The scale is the
  // geometric mean of the root magnitudes, which for a monic polynomial is
  // |p_n|^{1/n}: it centers the roots around 1 in log-space rather than
  // compressing the smallest roots below the solver's absolute tolerances
  // (which is what max-based scaling would do). Power-of-two scaling is
  // exact, so no precision is lost. The threshold is conservative: ordinary
  // inputs are left bit-identical.
  if (polynomial_.size() > 1) {
    const double constant_term = polynomial_(polynomial_.size() - 1);
    const int n_deg = static_cast<int>(polynomial_.size()) - 1;
    const int k_est = static_cast<int>(std::llround(
        static_cast<double>(std::ilogb(std::abs(constant_term))) / n_deg));
    constexpr int kScaleThreshold = 32;
    if (std::abs(k_est) >= kScaleThreshold) {
      scale_exponent_ = k_est;
      for (Eigen::Index i = 1; i < polynomial_.size(); ++i) {
        polynomial_(i) =
            std::ldexp(polynomial_(i), -static_cast<int>(i) * k_est);
      }
    }
  }

  original_polynomial_ = polynomial_;

  // Choose the initial starting value for the root-finding on the complex
  // plane.
  const double kDegToRad = M_PI / 180.0;
  double phi = 49.0 * kDegToRad;

  // Iterate until the polynomial has been completely deflated.
  for (int i = 0; i < degree; i++) {
    // Solve in closed form if the polynomial is small enough.
    if (polynomial_.size() <= 3) {
      break;
    }

    // Compute the root radius.
    const double root_radius = ComputeRootRadius();

    // Stage 1: Apply zero-shifts to the K-polynomial to separate the small
    // zeros of the polynomial.
    ApplyZeroShiftToKPolynomial(kNumZeroShiftIterations);

    // Stages 2 + 3: Apply fixed shift iterations to the K-polynomial to
    // separate the roots further, then find the root(s) with variable shift
    // iterations. If the variable shift stage fails (e.g. for tightly
    // clustered roots where the iteration stalls from a particular starting
    // angle), restart with a rotated starting point rather than giving up:
    // the K-polynomial retains the progress of previous attempts, so
    // subsequent restarts search from a different angle with a
    // better-separated K-sequence. This mirrors the restart structure of the
    // original RPOLY, which never lets a single stage-3 failure abort the
    // solve.
    bool root_found = false;
    for (int j = 0; j < kMaxFixedShiftRestarts; j++) {
      // The Cauchy bound is a valid but sometimes very loose lower bound on
      // the smallest root magnitude (e.g. quartics with a dominant middle
      // coefficient can have bound ~1 for roots at ~5). Restarts therefore
      // explore radius as well as angle: every fourth restart grows the
      // radius by 1.5x, covering a ~9x range across the restart budget.
      // (Clusters of roots far BELOW the Cauchy radius are handled by the
      // small-root rescue at exhaustion, so no downward sweep is needed.)
      const double radius = root_radius * std::pow(1.5, j / 4);
      const std::complex<double> root =
          radius * std::complex<double>(std::cos(phi), std::sin(phi));
      // Grow the fixed-shift budget across restarts (as in the original
      // RPOLY): later restarts are only reached for hard root
      // configurations, where more Stage-2 iterations are exactly what is
      // needed for the sigma/t sequences to settle.
      const ConvergenceType convergence = ApplyFixedShiftToKPolynomial(
          root, kFixedShiftIterationMultiplier * (i + 1) *
                    std::min(1 + j, 5));

      if (convergence != NO_CONVERGENCE) {
        if (ApplyVariableShiftToKPolynomial(convergence, root)) {
          root_found = true;
          break;
        }
        // Variable shift failed from this starting angle. Reset the
        // linear/quadratic attempt latches so the next restart gets a full
        // set of options.
        attempted_linear_shift_ = false;
        attempted_quadratic_shift_ = false;
      }

      // Rotate the initial root value on the complex plane and try again.
      phi += 94.0 * kDegToRad;
    }
    if (!root_found && !TrySmallRootRescue()) {
      return false;
    }
  }
  const bool success = SolveClosedFormPolynomial();
  if (success) {
    PolishRoots();
    if (scale_exponent_ != 0) {
      for (int r = 0; r < degree; ++r) {
        if (real_roots_ != NULL) {
          (*real_roots_)(r) = std::ldexp((*real_roots_)(r), scale_exponent_);
        }
        if (complex_roots_ != NULL) {
          (*complex_roots_)(r) =
              std::ldexp((*complex_roots_)(r), scale_exponent_);
        }
      }
    }
  }
  return success;
}

// Stage 1: Generate K-polynomials with no shifts (i.e. zero-shifts).
template <typename Vec>
void JenkinsTraubSolver<Vec>::ApplyZeroShiftToKPolynomial(
    const int num_iterations) {
  // K0 is the first order derivative of polynomial (scale is irrelevant to
  // the algorithm; kept for parity with the original).
  const Eigen::Index deg = polynomial_.size() - 1;
  k_polynomial_.resize(deg);
  const double inv_n = 1.0 / static_cast<double>(polynomial_.size());
  for (Eigen::Index i = 0; i < deg; ++i) {
    k_polynomial_(i) = static_cast<double>(deg - i) * polynomial_(i) * inv_n;
  }
  for (int i = 1; i < num_iterations; i++) {
    ComputeZeroShiftKPolynomial();
  }
}

template <typename Vec>
ConvergenceType JenkinsTraubSolver<Vec>::ApplyFixedShiftToKPolynomial(
    const std::complex<double>& root, const int max_iterations) {
  // Compute the fixed-shift quadratic:
  // sigma(z) = (x - m - n * i) * (x - m + n * i) = x^2 - 2 * m + m^2 + n^2.
  sigma_(0) = 1.0;
  sigma_(1) = -2.0 * root.real();
  sigma_(2) = root.real() * root.real() + root.imag() * root.imag();

  // Compute the quotient and remainder for divinding P by the quadratic
  // divisor. Since this iteration involves a fixed-shift sigma these may be
  // computed once prior to any iterations.
  Vec fs_polynomial_quotient, fs_k_polynomial_quotient;
  Eigen::Vector2d fs_remainder;
  QuadraticSyntheticDivision(
      polynomial_, sigma_, &fs_polynomial_quotient, &fs_remainder);

  // Compute a and b from the above equations.
  b_ = fs_remainder(0);
  a_ = fs_remainder(1) - b_ * sigma_(1);

  // Precompute P(s) for later using the equation above.
  const std::complex<double> p_at_root = a_ - b_ * std::conj(root);

  // These two containers hold values that we test for convergence such that the
  // zero index is the convergence value from 2 iterations ago, the first
  // index is from one iteration ago, and the second index is the current value.
  Vector3cd t_lambda = Vector3cd::Zero();
  Vector3d sigma_lambda = Vector3d::Zero();
  for (int i = 0; i < max_iterations; i++) {
    SafeMonicNormalize(&k_polynomial_);

    // Divide the shifted polynomial by the quadratic polynomial.
    QuadraticSyntheticDivision(
        k_polynomial_, sigma_, &fs_k_polynomial_quotient, &fs_remainder);
    d_ = fs_remainder(0);
    c_ = fs_remainder(1) - d_ * sigma_(1);

    // Test for convergence.
    const Vector3d variable_shift_sigma = ComputeNextSigma();
    const std::complex<double> k_at_root = c_ - d_ * std::conj(root);

    t_lambda.head<2>() = t_lambda.tail<2>().eval();
    sigma_lambda.head<2>() = sigma_lambda.tail<2>().eval();
    t_lambda(2) = root - p_at_root / k_at_root;
    sigma_lambda(2) = variable_shift_sigma(2);

    // Return with the convergence code if the sequence has converged.
    if (HasConverged(sigma_lambda)) {
      return QUADRATIC_CONVERGENCE;
    } else if (HasConverged(t_lambda)) {
      return LINEAR_CONVERGENCE;
    }

    // Compute K_next using the formula above.
    UpdateKPolynomialWithQuadraticShift(fs_polynomial_quotient,
                                        fs_k_polynomial_quotient);
  }

  return NO_CONVERGENCE;
}

// When a polynomial has a cluster of roots much smaller in magnitude than
// the rest, its trailing coefficients are correspondingly tiny and the
// trailing quadratic a_{n-2} x^2 + a_{n-1} x + a_n approximates the two
// smallest roots with relative error O(|root| * |a_{n-3}| / |a_{n-2}|) --
// comfortably inside Newton's basin. This rescue only runs when every
// fixed-shift restart has failed, and acceptance is protected by the same
// backward-error gate as the Ward path, so it can never degrade quality:
// either the polished seed certifies as a root, or the solve fails exactly
// as it would have anyway.
template <typename Vec>
bool JenkinsTraubSolver<Vec>::TrySmallRootRescue() {
  const Eigen::Index n = polynomial_.size();
  if (n < 4 || polynomial_(n - 3) == 0.0) {
    return false;
  }
  std::complex<double> seeds[2];
  FindQuadraticPolynomialRoots(polynomial_(n - 3), polynomial_(n - 2),
                               polynomial_(n - 1), seeds);
  for (int s = 0; s < 2; ++s) {
    std::complex<double> z = seeds[s];
    // Guarded Newton on the current (deflated) polynomial.
    for (int it = 0; it < 12; ++it) {
      std::complex<double> p(0.0, 0.0), dp(0.0, 0.0);
      for (Eigen::Index ci = 0; ci < n; ++ci) {
        dp = dp * z + p;
        p = p * z + polynomial_(ci);
      }
      if (std::abs(dp) == 0.0) {
        break;
      }
      const std::complex<double> step = p / dp;
      z -= step;
      if (std::abs(step) <= 1e-15 * std::abs(z)) {
        break;
      }
    }
    if (!ResidualAcceptable(polynomial_, z)) {
      continue;
    }
    if (std::abs(z.imag()) <= 1e-10 * std::abs(z.real())) {
      // Real root: standard deflation.
      double eval = 0.0;
      SyntheticDivisionAndEvaluate(polynomial_, z.real(),
                                   &deflated_polynomial_, &eval);
      AddRootToOutput(z.real(), 0);
      polynomial_ = deflated_polynomial_;
    } else {
      // Conjugate pair: deflate by x^2 - 2*Re(z)*x + |z|^2.
      sigma_.resize(3);  // may not have been sized if no quadratic shift ran
      sigma_(0) = 1.0;
      sigma_(1) = -2.0 * z.real();
      sigma_(2) = std::norm(z);
      QuadraticSyntheticDivision(polynomial_, sigma_, &polynomial_quotient_,
                                 &polynomial_remainder_);
      AddRootToOutput(z.real(), z.imag());
      AddRootToOutput(z.real(), -z.imag());
      polynomial_ = polynomial_quotient_;
    }
    return true;
  }
  return false;
}

template <typename Vec>
bool JenkinsTraubSolver<Vec>::CertifyAndDeflateReal(double candidate_root) {
  std::complex<double> z(candidate_root, 0.0);
  if (!ResidualAcceptable(polynomial_, z)) {
    return false;
  }
  double eval = 0.0;
  SyntheticDivisionAndEvaluate(polynomial_, candidate_root,
                               &deflated_polynomial_, &eval);
  AddRootToOutput(candidate_root, 0);
  polynomial_ = deflated_polynomial_;
  return true;
}

template <typename Vec>
bool JenkinsTraubSolver<Vec>::CertifyAndDeflatePair(
    const double* candidate_sigma) {
  sigma_.resize(3);
  sigma_(0) = candidate_sigma[0];
  sigma_(1) = candidate_sigma[1];
  sigma_(2) = candidate_sigma[2];
  std::complex<double> roots[2];
  FindQuadraticPolynomialRoots(sigma_(0), sigma_(1), sigma_(2), roots);
  if (!ResidualAcceptable(polynomial_, roots[0])) {
    return false;
  }
  QuadraticSyntheticDivision(polynomial_, sigma_, &polynomial_quotient_,
                             &polynomial_remainder_);
  AddRootToOutput(roots[0].real(), roots[0].imag());
  AddRootToOutput(roots[1].real(), roots[1].imag());
  polynomial_ = polynomial_quotient_;
  return true;
}

// Both shift budgets exhausted without Ward convergence: certify the best
// tracked candidates with the exact residual gate. The gate makes this
// strictly safe -- a candidate either has machine-level backward error or
// the solve fails exactly as it would have anyway.
template <typename Vec>
bool JenkinsTraubSolver<Vec>::TryResidualFallback() {
  if (best_pair_metric_ < std::numeric_limits<double>::max() &&
      CertifyAndDeflatePair(best_pair_sigma_)) {
    return true;
  }
  if (best_real_metric_ < std::numeric_limits<double>::max() &&
      CertifyAndDeflateReal(best_real_root_)) {
    return true;
  }
  return false;
}

template <typename Vec>
bool JenkinsTraubSolver<Vec>::ApplyVariableShiftToKPolynomial(
    const ConvergenceType& fixed_shift_convergence,
    const std::complex<double>& root) {
  attempted_linear_shift_ = false;
  attempted_quadratic_shift_ = false;
  best_real_metric_ = std::numeric_limits<double>::max();
  best_pair_metric_ = std::numeric_limits<double>::max();

  if (fixed_shift_convergence == LINEAR_CONVERGENCE) {
    return ApplyLinearShiftToKPolynomial(root, kMaxLinearShiftIterations);
  } else if (fixed_shift_convergence == QUADRATIC_CONVERGENCE) {
    return ApplyQuadraticShiftToKPolynomial(root, kMaxQuadraticShiftIterations);
  }
  return false;
}

// Generate K-polynomials with variable-shifts. During variable shifts, the
// quadratic shift is computed as:
//                | K0(s1)  K0(s2)  z^2 |
//                | K1(s1)  K1(s2)    z |
//                | K2(s1)  K2(s2)    1 |
//    sigma(z) = __________________________
//                  | K1(s1)  K2(s1) |
//                  | K2(s1)  K2(s2) |
// Where K0, K1, and K2 are successive zero-shifts of the K-polynomial.
//
// The K-polynomial shifts are otherwise exactly the same as Stage 2 after
// accounting for a variable-shift sigma.
template <typename Vec>
bool JenkinsTraubSolver<Vec>::ApplyQuadraticShiftToKPolynomial(
    const std::complex<double>& root, const int max_iterations) {
  // Both shift budgets exhausted without Ward convergence: accept the best
  // residual candidate if it meets the residual gate (stricter than the
  // Ward-path acceptance, so quality cannot regress).
  if (attempted_quadratic_shift_) {
    return TryResidualFallback();
  }

  const double kTinyRelativeStep = 0.01;

  // Compute the fixed-shift quadratic:
  // sigma(z) = (x - m - n * i) * (x - m + n * i) = x^2 - 2 * m + m^2 + n^2.
  sigma_(0) = 1.0;
  sigma_(1) = -2.0 * root.real();
  sigma_(2) = root.real() * root.real() + root.imag() * root.imag();

  // These two containers hold values that we test for convergence such that the
  // zero index is the convergence value from 2 iterations ago, the first
  // index is from one iteration ago, and the second index is the current value.
  double poly_at_root(0), prev_poly_at_root(0), prev_v(0);
  bool tried_fixed_shifts = false;

  // These containers maintain a history of the predicted roots. The convergence
  // of the algorithm is determined by the convergence of the root value.
  RootHistory<std::complex<double> > roots1, roots2;
  roots1.push(root);
  roots2.push(std::conj(root));
  for (int i = 0; i < max_iterations; i++) {
    // Terminate if the root evaluation is within our tolerance. This will
    // return false if we do not have enough samples.
    if (HasRootConverged(roots1) && HasRootConverged(roots2) &&
        ResidualAcceptable(polynomial_, roots1.v[1])) {
      AddRootToOutput(roots1.v[1].real(), roots1.v[1].imag());
      AddRootToOutput(roots2.v[1].real(), roots2.v[1].imag());
      polynomial_ = polynomial_quotient_;
      return true;
    }

    QuadraticSyntheticDivision(
        polynomial_, sigma_, &polynomial_quotient_, &polynomial_remainder_);

    // Compute a and b from the above equations.
    b_ = polynomial_remainder_(0);
    a_ = polynomial_remainder_(1) - b_ * sigma_(1);

    std::complex<double> roots[2];
    FindQuadraticPolynomialRoots(sigma_(0), sigma_(1), sigma_(2), roots);

    // Check that the roots are close. If not, then try a linear shift.
    // Latch BEFORE jumping: without this, mutually firing stall/routing
    // jumps in the linear and quadratic shifts recurse into each other
    // without bound (stack overflow). A shift abandoned mid-episode must
    // not be re-entered fresh within the same variable-shift episode;
    // restarts reset the latches for each new episode.
    if (std::abs(std::abs(roots[0].real()) - std::abs(roots[1].real())) >
        kRootPairTolerance * std::abs(roots[1].real())) {
      attempted_quadratic_shift_ = true;
      return ApplyLinearShiftToKPolynomial(root, kMaxLinearShiftIterations);
    }

    // If the iteration is stalling at a root pair then apply a few fixed shift
    // iterations to help convergence.
    poly_at_root =
        std::abs(a_ - roots[0].real() * b_) + std::abs(roots[0].imag() * b_);
    // O(1) tracking of the best sigma seen, for the exhaustion fallback
    // (multiple/clustered roots where sigma's v-component oscillates between
    // competing quadratic factors and Ward's step test never fires). The
    // proxy is only used for candidate SELECTION; exact residual
    // certification happens once, at fallback time, so a proxy misrank can
    // cause a missed rescue but never a wrong root.
    if (poly_at_root < best_pair_metric_) {
      best_pair_metric_ = poly_at_root;
      best_pair_sigma_[0] = sigma_(0);
      best_pair_sigma_[1] = sigma_(1);
      best_pair_sigma_[2] = sigma_(2);
    }
    const double rel_step = std::abs((sigma_(2) - prev_v) / sigma_(2));
    if (!tried_fixed_shifts && rel_step < kTinyRelativeStep &&
        prev_poly_at_root > poly_at_root) {
      tried_fixed_shifts = true;
      ApplyFixedShiftToKPolynomial(roots[0], kInnerFixedShiftIterations);
    }

    // Divide the shifted polynomial by the quadratic polynomial.
    QuadraticSyntheticDivision(
        k_polynomial_, sigma_, &k_polynomial_quotient_, &k_polynomial_remainder_);
    d_ = k_polynomial_remainder_(0);
    c_ = k_polynomial_remainder_(1) - d_ * sigma_(1);

    prev_v = sigma_(2);
    sigma_ = ComputeNextSigma();

    // Compute K_next using the formula above.
    UpdateKPolynomialWithQuadraticShift(polynomial_quotient_,
                                        k_polynomial_quotient_);
    SafeMonicNormalize(&k_polynomial_);
    prev_poly_at_root = poly_at_root;

    // Save the roots for convergence testing.
    roots1.push(roots[0]);
    roots2.push(roots[1]);
  }

  attempted_quadratic_shift_ = true;
  return ApplyLinearShiftToKPolynomial(root, kMaxLinearShiftIterations);
}

// Generate K-Polynomials with variable-shifts that are linear. The shift is
// computed as:
//   K_next(z) = 1 / (z - s) * (K(z) - K(s) / P(s) * P(z))
//   s_next = s - P(s) / K_next(s)
template <typename Vec>
bool JenkinsTraubSolver<Vec>::ApplyLinearShiftToKPolynomial(
    const std::complex<double>& root, const int max_iterations) {
  if (attempted_linear_shift_) {
    return TryResidualFallback();
  }

  // Compute an initial guess for the root.
  double real_root = (root -
                      EvalPoly(polynomial_, root) /
                          EvalPoly(k_polynomial_, root)).real();

  double polynomial_at_root(0), k_polynomial_at_root(0);

  // This container maintains a history of the predicted roots. The convergence
  // of the algorithm is determined by the convergence of the root value.
  RootHistory<double> roots;
  roots.push(real_root);
  for (int i = 0; i < max_iterations; i++) {
    // Terminate if the root evaluation is within our tolerance. This will
    // return false if we do not have enough samples.
    if (HasRootConverged(roots) &&
        ResidualAcceptable(polynomial_, roots.v[1])) {
      AddRootToOutput(roots.v[1], 0);
      polynomial_ = deflated_polynomial_;
      return true;
    }

    const double prev_polynomial_at_root = polynomial_at_root;
    SyntheticDivisionAndEvaluate(
        polynomial_, real_root, &deflated_polynomial_, &polynomial_at_root);

    // O(1) tracking of the best iterate seen, for the exhaustion fallback
    // (near a multiple root the step sequence stalls at a linear rate long
    // before Ward's step tolerance is met, while |P(s)| reaches the
    // rounding floor quickly).
    if (std::abs(polynomial_at_root) < best_real_metric_) {
      best_real_metric_ = std::abs(polynomial_at_root);
      best_real_root_ = real_root;
    }
    // If the root is (numerically) exactly the root then end early. Otherwise,
    // the k polynomial will be filled with inf or nans. The absolute test is
    // only a cheap pre-filter: on its own it accepts nearly any point when
    // the roots are small (|P| ~ 1e-15 throughout a cluster of roots near
    // 1e-5), so the scale-aware residual gate must confirm it.
    if (std::abs(polynomial_at_root) <= kAbsoluteTolerance &&
        ResidualAcceptable(polynomial_, real_root)) {
      AddRootToOutput(real_root, 0);
      polynomial_ = deflated_polynomial_;
      return true;
    }

    // Update the K-Polynomial:
    //   K_next = deflated_K + scale * deflated_P, tail-aligned (deflated_K is
    //   one entry shorter than deflated_P). Fused, no temporaries.
    SyntheticDivisionAndEvaluate(k_polynomial_, real_root,
                                 &deflated_k_polynomial_, &k_polynomial_at_root);
    const double scale = -k_polynomial_at_root / polynomial_at_root;
    const Eigen::Index dp = deflated_polynomial_.size();
    k_polynomial_.resize(dp);
    k_polynomial_(0) = scale * deflated_polynomial_(0);
    for (Eigen::Index j = 1; j < dp; ++j) {
      k_polynomial_(j) =
          scale * deflated_polynomial_(j) + deflated_k_polynomial_(j - 1);
    }

    SafeMonicNormalize(&k_polynomial_);

    // Compute the update for the root estimation.
    k_polynomial_at_root = EvalPoly(k_polynomial_, real_root);
    const double delta_root = polynomial_at_root / k_polynomial_at_root;
    real_root -= polynomial_at_root / k_polynomial_at_root;

    // Save the root so that convergence can be measured. Only the 3 most
    // recently root values are needed.
    roots.push(real_root);

    // If the linear iterations appear to be stalling then we may have found a
    // double real root of the form (z - x^2). Attempt a quadratic variable
    // shift from the current estimate of the root.
    if (i >= 2 &&
        std::abs(delta_root) < 0.001 * std::abs(real_root) &&
        std::abs(prev_polynomial_at_root) < std::abs(polynomial_at_root)) {
      const std::complex<double> new_root(real_root, 0);
      // Latch before jumping (see the matching note in the quadratic
      // shift): prevents unbounded mutual recursion between the stall
      // heuristics of the two shifts.
      attempted_linear_shift_ = true;
      return ApplyQuadraticShiftToKPolynomial(new_root,
                                              kMaxQuadraticShiftIterations);
    }
  }

  attempted_linear_shift_ = true;
  return ApplyQuadraticShiftToKPolynomial(root, kMaxQuadraticShiftIterations);
}


template <typename Vec>
void JenkinsTraubSolver<Vec>::PolishRoots() {
  if (real_roots_ == NULL || complex_roots_ == NULL) {
    return;
  }
  const Eigen::Index n = original_polynomial_.size();
  if (n < 3) {
    return;
  }
  for (int r = 0; r < num_solved_roots_; ++r) {
    if ((*complex_roots_)(r) == 0.0) {
      (*real_roots_)(r) = internal::PolishRoot(original_polynomial_,
                                               (*real_roots_)(r));
      continue;
    }
    // Complex root. Conjugate pairs are emitted consecutively; polish the
    // first of the two and mirror, so the pair stays exactly conjugate.
    const std::complex<double> z0((*real_roots_)(r), (*complex_roots_)(r));
    const bool is_pair =
        r + 1 < num_solved_roots_ &&
        (*real_roots_)(r + 1) == z0.real() &&
        (*complex_roots_)(r + 1) == -z0.imag();
    const std::complex<double> z =
        internal::PolishRoot(original_polynomial_, z0);
    (*real_roots_)(r) = z.real();
    (*complex_roots_)(r) = z.imag();
    if (is_pair) {
      (*real_roots_)(r + 1) = z.real();
      (*complex_roots_)(r + 1) = -z.imag();
      ++r;
    }
  }
}

template <typename Vec>
void JenkinsTraubSolver<Vec>::AddRootToOutput(const double real, const double imag) {
  if (real_roots_ != NULL) {
    (*real_roots_)(num_solved_roots_) = real;
  }
  if (complex_roots_ != NULL) {
    (*complex_roots_)(num_solved_roots_) = imag;
  }
  ++num_solved_roots_;
}

template <typename Vec>
void JenkinsTraubSolver<Vec>::RemoveZeroRoots() {
  int num_zero_roots = 0;

  const Eigen::Index last = polynomial_.size() - 1;
  while (polynomial_(last - num_zero_roots) == 0) {
    ++num_zero_roots;
  }

  // The output roots have 0 as the default value so there is no need to
  // explicitly add the zero roots.
  polynomial_.conservativeResize(polynomial_.size() - num_zero_roots);
}

template <typename Vec>
bool JenkinsTraubSolver<Vec>::SolveClosedFormPolynomial() {
  const int degree = polynomial_.size() - 1;

  // Is the polynomial constant?
  if (degree == 0) {
    std::cout << "Trying to extract roots from a constant "
              << "polynomial in FindPolynomialRoots" << std::endl;
    // We return true with no roots, not false, as if the polynomial is constant
    // it is correct that there are no roots. It is not the case that they were
    // there, but that we have failed to extract them.
    return true;
  }

  // Linear
  if (degree == 1) {
    AddRootToOutput(FindLinearPolynomialRoots(polynomial_(0), polynomial_(1)),
                    0);
    return true;
  }

  // Quadratic
  if (degree == 2) {
    std::complex<double> roots[2];
    FindQuadraticPolynomialRoots(polynomial_(0), polynomial_(1), polynomial_(2),
                                 roots);
    AddRootToOutput(roots[0].real(), roots[0].imag());
    AddRootToOutput(roots[1].real(), roots[1].imag());
    return true;
  }

  return false;
}

// Computes a lower bound on the radius of the roots of polynomial by examining
// the Cauchy sequence:
//
//    z^n + |a_1| * z^{n - 1} + ... + |a_{n-1}| * z - |a_n|
//
// The unique positive zero of this polynomial is an approximate lower bound of
// the radius of zeros of the original polynomial.
template <typename Vec>
double JenkinsTraubSolver<Vec>::ComputeRootRadius() {
  constexpr double kEpsilon = 1e-2;
  constexpr int kMaxIterations = 100;

  // Cauchy polynomial: |a_0| x^n + ... + |a_{n-1}| x - |a_n|. Solve for its
  // unique positive zero with Newton, evaluating P and P' in one fused pass.
  const Eigen::Index n = polynomial_.size();
  // Start at 1, or for small roots at ~the geometric mean of the root
  // magnitudes, |a_n / a_0|^(1/deg), rounded to a power of two (ilogb/ldexp,
  // no pow). From 1, Newton descends to tiny roots only linearly (a factor of
  // ~(deg-1)/deg per step) and would hit kMaxIterations far above the root.
  double root = 1.0;
  const double ratio = std::abs(polynomial_(n - 1) / polynomial_(0));
  if (ratio > 0.0 && ratio < 1.0) {
    root = std::ldexp(1.0, std::ilogb(ratio) / static_cast<int>(n - 1));
  }
  for (int it = 0; it < kMaxIterations; ++it) {
    double p = std::abs(polynomial_(0));
    double dp = 0.0;
    for (Eigen::Index i = 1; i < n; ++i) {
      dp = dp * root + p;
      const double coeff =
          (i == n - 1) ? -std::abs(polynomial_(i)) : std::abs(polynomial_(i));
      p = p * root + coeff;
    }
    const double prev = root;
    root -= p / dp;
    // Relative below 1: an absolute 1e-2 stops at ~0.05 for roots ~1e-10.
    if (std::abs(prev - root) < kEpsilon * std::min(root, 1.0)) {
      break;
    }
  }
  return root;
}

// The k polynomial with a zero-shift is
//  (K(x) - K(0) / P(0) * P(x)) / x.
//
// This is equivalent to:
//    K(x) - K(0)      K(0)     P(x) - P(0)
//    ___________   -  ____  *  ___________
//         x           P(0)          x
//
// Note that removing the constant term and dividing by x is equivalent to
// shifting the polynomial to one degree lower in our representation.
template <typename Vec>
void JenkinsTraubSolver<Vec>::ComputeZeroShiftKPolynomial() {
  // Evaluating the polynomial at zero is equivalent to the constant term
  // (i.e. the last coefficient). Note that reverse() is an expression and does
  // not actually reverse the vector elements.
  const double polynomial_at_zero = polynomial_(polynomial_.size() - 1);
  const double k_at_zero = k_polynomial_(k_polynomial_.size() - 1);
  const double scale = -k_at_zero / polynomial_at_zero;

  // K has size n (one less than P). Result also has size n:
  //   K_next = scale * P.head(n) with K.head(n-1) added tail-aligned.
  const Eigen::Index n = polynomial_.size() - 1;
  // Shift K down by one degree in place (drop the constant term) and add
  // scale * P.head(n). Iterate from the low-order end to avoid clobbering.
  for (Eigen::Index i = n - 1; i > 0; --i) {
    k_polynomial_(i) = scale * polynomial_(i) + k_polynomial_(i - 1);
  }
  k_polynomial_(0) = scale * polynomial_(0);
}

// The iterations are computed with the following equation:
//              a^2 + u * a * b + v * b^2
//   K_next =  ___________________________ * Q_K
//                    b * c - a * d
//
//                      a * c + u * a * d + v * b * d
//             +  (z - _______________________________) * Q_P + b.
//                              b * c - a * d
//
// This is done using *only* realy arithmetic so it can be done very fast!
template <typename Vec>
void JenkinsTraubSolver<Vec>::UpdateKPolynomialWithQuadraticShift(
    const Vec& polynomial_quotient,
    const Vec& k_polynomial_quotient) {
  const double denom = b_ * c_ - a_ * d_;
  const double coefficient_q_k =
      (a_ * a_ + sigma_(1) * a_ * b_ + sigma_(2) * b_ * b_) / denom;
  const double t =
      -(a_ * c_ + sigma_(1) * a_ * d_ + sigma_(2) * b_ * d_) / denom;

  // K_next = coefficient_q_k * Q_K + (z + t) * Q_P + b, fused in one pass.
  // Q_P has size m; (z + t) * Q_P has size m + 1; Q_K has size m - 1 and is
  // tail-aligned (low-order) against it.
  const Eigen::Index m = polynomial_quotient.size();  // Q_K has size m - 1.
  k_polynomial_.resize(m + 1);
  k_polynomial_(0) = polynomial_quotient(0);
  k_polynomial_(1) = polynomial_quotient(1) + t * polynomial_quotient(0);
  for (Eigen::Index i = 2; i < m; ++i) {
    k_polynomial_(i) = coefficient_q_k * k_polynomial_quotient(i - 2) +
                       polynomial_quotient(i) + t * polynomial_quotient(i - 1);
  }
  k_polynomial_(m) =
      coefficient_q_k * k_polynomial_quotient(m - 2) +
      t * polynomial_quotient(m - 1) + b_;
}

// Using a bit of algebra, the update of sigma(z) can be computed from the
// previous value along with a, b, c, and d defined above. The details of this
// simplification can be found in "Three Stage Variable-Shift Iterations for the
// Solution of Polynomial Equations With a Posteriori Error Bounds for the
// Zeros" by M.A. Jenkins, Doctoral Thesis, Stanford Univeristy, 1969.
//
// NOTE: we assume the leading term of quadratic_sigma is 1.0.
template <typename Vec>
Vector3d JenkinsTraubSolver<Vec>::ComputeNextSigma() {
  const double u = sigma_(1);
  const double v = sigma_(2);

  const Eigen::Index kn = k_polynomial_.size();
  const Eigen::Index pn = polynomial_.size();
  const double b1 = -k_polynomial_(kn - 1) / polynomial_(pn - 1);
  const double b2 = -(k_polynomial_(kn - 2) + b1 * polynomial_(pn - 2)) /
                    polynomial_(pn - 1);

  const double a1 = b_* c_ - a_ * d_;
  const double a2 = a_ * c_ + u * a_ * d_ + v * b_* d_;
  const double c2 = b1 * a2;
  const double c3 = b1 * b1 * (a_ * a_ + u * a_ * b_ + v * b_ * b_);
  const double c4 = v * b2 * a1 - c2 - c3;
  const double c1 = c_ * c_ + u * c_ * d_ + v * d_ * d_ +
                    b1 * (a_ * c_ + u * b_ * c_ + v * b_ * d_) - c4;
  const double delta_u = -(u * (c2 + c3) + v * (b1 * a1 + b2 * a2)) / c1;
  const double delta_v = v * c4 / c1;

  // Update u and v in the quadratic sigma.
  Vector3d new_quadratic_sigma;
  new_quadratic_sigma(0) = 1.0;
  new_quadratic_sigma(1) = u + delta_u;
  new_quadratic_sigma(2) = v + delta_v;
  return new_quadratic_sigma;
}

}  // namespace

bool FindPolynomialRootsJenkinsTraub(const VectorXd& polynomial,
                                     VectorXd* real_roots,
                                     VectorXd* complex_roots) {
  if (polynomial.size() <= kStackCoeffs) {
    // Entirely stack-resident: zero heap allocations per solve.
    StackVec coeffs = polynomial;
    JenkinsTraubSolver<StackVec> solver(coeffs, real_roots, complex_roots);
    return solver.ExtractRoots();
  }
  JenkinsTraubSolver<VectorXd> solver(polynomial, real_roots, complex_roots);
  return solver.ExtractRoots();
}

}  // namespace rpoly_plus_plus
