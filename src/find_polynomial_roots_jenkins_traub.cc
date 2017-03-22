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

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "src/polynomial.h"

namespace rpoly_plus_plus {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Vector3cd;
using Eigen::VectorXcd;

namespace {

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// Machine precision constants.
static const double mult_eps = std::numeric_limits<double>::epsilon();
static const double sum_eps = std::numeric_limits<double>::epsilon();
static const double kAbsoluteTolerance = 1e-14;
static const double kRelativeTolerance = 1e-10;

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
  const double D = b * b - 4 * a * c;
  const double sqrt_D = std::sqrt(std::abs(D));

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
void SyntheticDivisionAndEvaluate(const VectorXd& polynomial,
                                  const double x,
                                  VectorXd* quotient,
                                  double* eval) {
  quotient->setZero(polynomial.size() - 1);
  (*quotient)(0) = polynomial(0);
  for (int i = 1; i < polynomial.size() - 1; i++) {
    (*quotient)(i) = polynomial(i) + (*quotient)(i - 1) * x;
  }
  const VectorXd::ReverseReturnType& creverse_quotient = quotient->reverse();
  *eval = polynomial.reverse()(0) + creverse_quotient(0) * x;
}

// Perform division of a polynomial by a quadratic factor. The quadratic divisor
// should have leading 1s.
void QuadraticSyntheticDivision(const VectorXd& polynomial,
                                const VectorXd& quadratic_divisor,
                                VectorXd* quotient,
                                VectorXd* remainder) {
  quotient->setZero(polynomial.size() - 2);
  remainder->setZero(2);

  (*quotient)(0) = polynomial(0);

  // If the quotient is a constant then polynomial is degree 2 and the math is
  // simple.
  if (quotient->size() == 1) {
    *remainder =
        polynomial.tail<2>() - polynomial(0) * quadratic_divisor.tail<2>();
    return;
  }

  (*quotient)(1) = polynomial(1) - polynomial(0) * quadratic_divisor(1);
  for (int i = 2; i < polynomial.size() - 2; i++) {
    (*quotient)(i) = polynomial(i) - (*quotient)(i - 2) * quadratic_divisor(2) -
                     (*quotient)(i - 1) * quadratic_divisor(1);
  }

  const VectorXd::ReverseReturnType &creverse_quotient = quotient->reverse();
  (*remainder)(0) = polynomial.reverse()(1) -
                    quadratic_divisor(1) * creverse_quotient(0) -
                    quadratic_divisor(2) * creverse_quotient(1);
  (*remainder)(1) =
      polynomial.reverse()(0) - quadratic_divisor(2) * creverse_quotient(0);
}

// Determines whether the iteration has converged by examining the three most
// recent values for convergence.
template<typename T>
bool HasConverged(const T& sequence) {
  const bool convergence_condition_1 =
      std::abs(sequence(1) - sequence(0)) < std::abs(sequence(0)) / 2.0;
  const bool convergence_condition_2 =
      std::abs(sequence(2) - sequence(1)) < std::abs(sequence(1)) / 2.0;

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
template <typename T>
bool HasRootConverged(const std::vector<T>& roots) {
  static const double kRootMagnitudeTolerance = 1e-8;
  if (roots.size() != 3) {
    return false;
  }

  const double e_i = std::abs(roots[2] - roots[1]);
  const double e_i_minus_1 = std::abs(roots[1] - roots[0]);
  const double mag_root = std::abs(roots[1]);
  if (e_i <= e_i_minus_1) {
    if (mag_root < kRootMagnitudeTolerance) {
      return e_i < kAbsoluteTolerance;
    } else {
      return e_i / mag_root <= kRelativeTolerance;
    }
  }

  return false;
}

// Implementation closely follows the three-stage algorithm for finding roots of
// polynomials with real coefficients as outlined in: "A Three-Stage Algorithm
// for Real Polynomaials Using Quadratic Iteration" by Jenkins and Traub, SIAM
// 1970. Please note that this variant is different than the complex-coefficient
// version, and is estimated to be up to 4 times faster.
class JenkinsTraubSolver {
 public:
  JenkinsTraubSolver(const VectorXd& coeffs,
                     VectorXd* real_roots,
                     VectorXd* complex_roots)
      : polynomial_(coeffs),
        real_roots_(real_roots),
        complex_roots_(complex_roots),
        num_solved_roots_(0) {}

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
  VectorXd ComputeNextSigma();

  // Updates the K-polynomial based on the current value of sigma for the fixed
  // or variable shift stage.
  void UpdateKPolynomialWithQuadraticShift(
      const VectorXd& polynomial_quotient,
      const VectorXd& k_polynomial_quotient);

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

  // Helper variables to manage the polynomials as they are being manipulated
  // and deflated.
  VectorXd polynomial_;
  VectorXd k_polynomial_;
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
  int num_solved_roots_;

  // Keeps track of whether the linear and quadratic shifts have been attempted
  // yet so that we do not attempt the same shift twice.
  bool attempted_linear_shift_;
  bool attempted_quadratic_shift_;

  // Number of zero-shift iterations to perform.
  static const int kNumZeroShiftIterations = 20;

  // The number of fixed shift iterations is computed as
  //   # roots found * this multiplier.
  static const int kFixedShiftIterationMultiplier = 20;

  // If the fixed shift iterations fail to converge, we restart this many times
  // before considering the solve attempt as a failure.
  static const int kMaxFixedShiftRestarts = 20;

  // The maximum number of linear shift iterations to perform before considering
  // the shift as a failure.
  static const int kMaxLinearShiftIterations = 20;

  // The maximum number of quadratic shift iterations to perform before
  // considering the shift as a failure.
  static const int kMaxQuadraticShiftIterations = 20;

  // When quadratic shift iterations are stalling, we attempt a few fixed shift
  // iterations to help convergence.
  static const int kInnerFixedShiftIterations = 5;

  // During quadratic iterations, the real values of the root pairs should be
  // nearly equal since the root pairs are complex conjugates. This tolerance
  // measures how much the real values may diverge before consider the quadratic
  // shift to be failed.
  static constexpr double kRootPairTolerance = 0.01;
};

bool JenkinsTraubSolver::ExtractRoots() {
  if (polynomial_.size() == 0) {
    std::cout << "Invalid polynomial of size 0 passed to "
                 "FindPolynomialRootsJenkinsTraub" << std::endl;
    return false;
  }

  // Remove any leading zeros of the polynomial.
  polynomial_ = RemoveLeadingZeros(polynomial_);
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

  // Choose the initial starting value for the root-finding on the complex
  // plane.
  const double kDegToRad = M_PI / 180.0;
  double phi = 49.0 * kDegToRad;

  // Iterate until the polynomial has been completely deflated.
  for (int i = 0; i < degree; i++) {
    // Compute the root radius.
    const double root_radius = ComputeRootRadius();

    // Solve in closed form if the polynomial is small enough.
    if (polynomial_.size() <= 3) {
      break;
    }

    // Stage 1: Apply zero-shifts to the K-polynomial to separate the small
    // zeros of the polynomial.
    ApplyZeroShiftToKPolynomial(kNumZeroShiftIterations);

    // Stage 2: Apply fixed shift iterations to the K-polynomial to separate the
    // roots further.
    std::complex<double> root;
    ConvergenceType convergence = NO_CONVERGENCE;
    for (int j = 0; j < kMaxFixedShiftRestarts; j++) {
      root = root_radius * std::complex<double>(std::cos(phi), std::sin(phi));
      convergence = ApplyFixedShiftToKPolynomial(
          root, kFixedShiftIterationMultiplier * (i + 1));
      if (convergence != NO_CONVERGENCE) {
        break;
      }

      // Rotate the initial root value on the complex plane and try again.
      phi += 94.0 * kDegToRad;
    }

    // Stage 3: Find the root(s) with variable shift iterations on the
    // K-polynomial. If this stage was not successful then we return a failure.
    if (!ApplyVariableShiftToKPolynomial(convergence, root)) {
      return false;
    }
  }
  return SolveClosedFormPolynomial();
}

// Stage 1: Generate K-polynomials with no shifts (i.e. zero-shifts).
void JenkinsTraubSolver::ApplyZeroShiftToKPolynomial(
    const int num_iterations) {
  // K0 is the first order derivative of polynomial.
  k_polynomial_ = DifferentiatePolynomial(polynomial_) / polynomial_.size();
  for (int i = 1; i < num_iterations; i++) {
    ComputeZeroShiftKPolynomial();
  }
}

ConvergenceType JenkinsTraubSolver::ApplyFixedShiftToKPolynomial(
    const std::complex<double>& root, const int max_iterations) {
  // Compute the fixed-shift quadratic:
  // sigma(z) = (x - m - n * i) * (x - m + n * i) = x^2 - 2 * m + m^2 + n^2.
  sigma_(0) = 1.0;
  sigma_(1) = -2.0 * root.real();
  sigma_(2) = root.real() * root.real() + root.imag() * root.imag();

  // Compute the quotient and remainder for divinding P by the quadratic
  // divisor. Since this iteration involves a fixed-shift sigma these may be
  // computed once prior to any iterations.
  VectorXd polynomial_quotient, polynomial_remainder;
  QuadraticSyntheticDivision(
      polynomial_, sigma_, &polynomial_quotient, &polynomial_remainder);

  // Compute a and b from the above equations.
  b_ = polynomial_remainder(0);
  a_ = polynomial_remainder(1) - b_ * sigma_(1);

  // Precompute P(s) for later using the equation above.
  const std::complex<double> p_at_root = a_ - b_ * std::conj(root);

  // These two containers hold values that we test for convergence such that the
  // zero index is the convergence value from 2 iterations ago, the first
  // index is from one iteration ago, and the second index is the current value.
  Vector3cd t_lambda = Vector3cd::Zero();
  Vector3d sigma_lambda = Vector3d::Zero();
  VectorXd k_polynomial_quotient, k_polynomial_remainder;
  for (int i = 0; i < max_iterations; i++) {
    k_polynomial_ /= k_polynomial_(0);

    // Divide the shifted polynomial by the quadratic polynomial.
    QuadraticSyntheticDivision(
        k_polynomial_, sigma_, &k_polynomial_quotient, &k_polynomial_remainder);
    d_ = k_polynomial_remainder(0);
    c_ = k_polynomial_remainder(1) - d_ * sigma_(1);

    // Test for convergence.
    const VectorXd variable_shift_sigma = ComputeNextSigma();
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
    UpdateKPolynomialWithQuadraticShift(polynomial_quotient,
                                        k_polynomial_quotient);
  }

  return NO_CONVERGENCE;
}

bool JenkinsTraubSolver::ApplyVariableShiftToKPolynomial(
    const ConvergenceType& fixed_shift_convergence,
    const std::complex<double>& root) {
  attempted_linear_shift_ = false;
  attempted_quadratic_shift_ = false;

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
bool JenkinsTraubSolver::ApplyQuadraticShiftToKPolynomial(
    const std::complex<double>& root, const int max_iterations) {
  // Only proceed if we have not already tried a quadratic shift.
  if (attempted_quadratic_shift_) {
    return false;
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
  VectorXd polynomial_quotient, polynomial_remainder, k_polynomial_quotient,
      k_polynomial_remainder;
  double poly_at_root(0), prev_poly_at_root(0), prev_v(0);
  bool tried_fixed_shifts = false;

  // These containers maintain a history of the predicted roots. The convergence
  // of the algorithm is determined by the convergence of the root value.
  std::vector<std::complex<double> > roots1, roots2;
  roots1.push_back(root);
  roots2.push_back(std::conj(root));
  for (int i = 0; i < max_iterations; i++) {
    // Terminate if the root evaluation is within our tolerance. This will
    // return false if we do not have enough samples.
    if (HasRootConverged(roots1) && HasRootConverged(roots2)) {
      AddRootToOutput(roots1[1].real(), roots1[1].imag());
      AddRootToOutput(roots2[1].real(), roots2[1].imag());
      polynomial_ = polynomial_quotient;
      return true;
    }

    QuadraticSyntheticDivision(
        polynomial_, sigma_, &polynomial_quotient, &polynomial_remainder);

    // Compute a and b from the above equations.
    b_ = polynomial_remainder(0);
    a_ = polynomial_remainder(1) - b_ * sigma_(1);

    std::complex<double> roots[2];
    FindQuadraticPolynomialRoots(sigma_(0), sigma_(1), sigma_(2), roots);

    // Check that the roots are close. If not, then try a linear shift.
    if (std::abs(std::abs(roots[0].real()) - std::abs(roots[1].real())) >
        kRootPairTolerance * std::abs(roots[1].real())) {
      return ApplyLinearShiftToKPolynomial(root, kMaxLinearShiftIterations);
    }

    // If the iteration is stalling at a root pair then apply a few fixed shift
    // iterations to help convergence.
    poly_at_root =
        std::abs(a_ - roots[0].real() * b_) + std::abs(roots[0].imag() * b_);
    const double rel_step = std::abs((sigma_(2) - prev_v) / sigma_(2));
    if (!tried_fixed_shifts && rel_step < kTinyRelativeStep &&
        prev_poly_at_root > poly_at_root) {
      tried_fixed_shifts = true;
      ApplyFixedShiftToKPolynomial(roots[0], kInnerFixedShiftIterations);
    }

    // Divide the shifted polynomial by the quadratic polynomial.
    QuadraticSyntheticDivision(
        k_polynomial_, sigma_, &k_polynomial_quotient, &k_polynomial_remainder);
    d_ = k_polynomial_remainder(0);
    c_ = k_polynomial_remainder(1) - d_ * sigma_(1);

    prev_v = sigma_(2);
    sigma_ = ComputeNextSigma();

    // Compute K_next using the formula above.
    UpdateKPolynomialWithQuadraticShift(polynomial_quotient,
                                        k_polynomial_quotient);
    k_polynomial_ /= k_polynomial_(0);
    prev_poly_at_root = poly_at_root;

    // Save the roots for convergence testing.
    roots1.push_back(roots[0]);
    roots2.push_back(roots[1]);
    if (roots1.size() > 3) {
      roots1.erase(roots1.begin());
      roots2.erase(roots2.begin());
    }
  }

  attempted_quadratic_shift_ = true;
  return ApplyLinearShiftToKPolynomial(root, kMaxLinearShiftIterations);
}

// Generate K-Polynomials with variable-shifts that are linear. The shift is
// computed as:
//   K_next(z) = 1 / (z - s) * (K(z) - K(s) / P(s) * P(z))
//   s_next = s - P(s) / K_next(s)
bool JenkinsTraubSolver::ApplyLinearShiftToKPolynomial(
    const std::complex<double>& root, const int max_iterations) {
  if (attempted_linear_shift_) {
    return false;
  }

  // Compute an initial guess for the root.
  double real_root = (root -
                      EvaluatePolynomial(polynomial_, root) /
                          EvaluatePolynomial(k_polynomial_, root)).real();

  VectorXd deflated_polynomial, deflated_k_polynomial;
  double polynomial_at_root(0), k_polynomial_at_root(0);

  // This container maintains a history of the predicted roots. The convergence
  // of the algorithm is determined by the convergence of the root value.
  std::vector<double> roots;
  roots.push_back(real_root);;
  for (int i = 0; i < max_iterations; i++) {
    // Terminate if the root evaluation is within our tolerance. This will
    // return false if we do not have enough samples.
    if (HasRootConverged(roots)) {
      AddRootToOutput(roots[1], 0);
      polynomial_ = deflated_polynomial;
      return true;
    }

    const double prev_polynomial_at_root = polynomial_at_root;
    SyntheticDivisionAndEvaluate(
        polynomial_, real_root, &deflated_polynomial, &polynomial_at_root);

    // If the root is exactly the root then end early. Otherwise, the k
    // polynomial will be filled with inf or nans.
    if (std::abs(polynomial_at_root) <= kAbsoluteTolerance) {
      AddRootToOutput(real_root, 0);
      polynomial_ = deflated_polynomial;
      return true;
    }

    // Update the K-Polynomial.
    SyntheticDivisionAndEvaluate(k_polynomial_, real_root,
                                 &deflated_k_polynomial, &k_polynomial_at_root);
    k_polynomial_ = AddPolynomials(
        deflated_k_polynomial,
        -k_polynomial_at_root / polynomial_at_root * deflated_polynomial);

    k_polynomial_ /= k_polynomial_(0);

    // Compute the update for the root estimation.
    k_polynomial_at_root = EvaluatePolynomial(k_polynomial_, real_root);
    const double delta_root = polynomial_at_root / k_polynomial_at_root;
    real_root -= polynomial_at_root / k_polynomial_at_root;

    // Save the root so that convergence can be measured. Only the 3 most
    // recently root values are needed.
    roots.push_back(real_root);
    if (roots.size() > 3) {
      roots.erase(roots.begin());
    }

    // If the linear iterations appear to be stalling then we may have found a
    // double real root of the form (z - x^2). Attempt a quadratic variable
    // shift from the current estimate of the root.
    if (i >= 2 &&
        std::abs(delta_root) < 0.001 * std::abs(real_root) &&
        std::abs(prev_polynomial_at_root) < std::abs(polynomial_at_root)) {
      const std::complex<double> new_root(real_root, 0);
      return ApplyQuadraticShiftToKPolynomial(new_root,
                                              kMaxQuadraticShiftIterations);
    }
  }

  attempted_linear_shift_ = true;
  return ApplyQuadraticShiftToKPolynomial(root, kMaxQuadraticShiftIterations);
}

void JenkinsTraubSolver::AddRootToOutput(const double real, const double imag) {
  if (real_roots_ != NULL) {
    (*real_roots_)(num_solved_roots_) = real;
  }
  if (complex_roots_ != NULL) {
    (*complex_roots_)(num_solved_roots_) = imag;
  }
  ++num_solved_roots_;
}

void JenkinsTraubSolver::RemoveZeroRoots() {
  int num_zero_roots = 0;

  const VectorXd::ReverseReturnType& creverse_polynomial =
      polynomial_.reverse();
  while (creverse_polynomial(num_zero_roots) == 0) {
    ++num_zero_roots;
  }

  // The output roots have 0 as the default value so there is no need to
  // explicitly add the zero roots.
  polynomial_ = polynomial_.head(polynomial_.size() - num_zero_roots).eval();
}

bool JenkinsTraubSolver::SolveClosedFormPolynomial() {
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
double JenkinsTraubSolver::ComputeRootRadius() {
  static const double kEpsilon = 1e-2;
  static const int kMaxIterations = 100;

  VectorXd poly = polynomial_;
  // Take the absolute value of all coefficients.
  poly = poly.array().abs();
  // Negate the last coefficient.
  poly.reverse()(0) *= -1.0;

  // Find the unique positive zero using Newton-Raphson iterations.
  double x0 = 1.0;
  return FindRootIterativeNewton(poly, x0, kEpsilon, kMaxIterations);
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
void JenkinsTraubSolver::ComputeZeroShiftKPolynomial() {
  // Evaluating the polynomial at zero is equivalent to the constant term
  // (i.e. the last coefficient). Note that reverse() is an expression and does
  // not actually reverse the vector elements.
  const double polynomial_at_zero = polynomial_(polynomial_.size() - 1);
  const double k_at_zero = k_polynomial_(k_polynomial_.size() - 1);

  k_polynomial_ = AddPolynomials(k_polynomial_.head(k_polynomial_.size() - 1),
                                 -k_at_zero / polynomial_at_zero *
                                     polynomial_.head(polynomial_.size() - 1));
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
void JenkinsTraubSolver::UpdateKPolynomialWithQuadraticShift(
    const VectorXd& polynomial_quotient,
    const VectorXd& k_polynomial_quotient) {
  const double coefficient_q_k =
      (a_ * a_ + sigma_(1) * a_ * b_ + sigma_(2) * b_ * b_) /
      (b_ * c_ - a_ * d_);
  VectorXd linear_polynomial(2);
  linear_polynomial(0) = 1.0;
  linear_polynomial(1) =
      -(a_ * c_ + sigma_(1) * a_ * d_ + sigma_(2) * b_ * d_) /
      (b_ * c_ - a_ * d_);
  k_polynomial_ = AddPolynomials(
      coefficient_q_k * k_polynomial_quotient,
      MultiplyPolynomials(linear_polynomial, polynomial_quotient));
  k_polynomial_(k_polynomial_.size() - 1) += b_;
}

// Using a bit of algebra, the update of sigma(z) can be computed from the
// previous value along with a, b, c, and d defined above. The details of this
// simplification can be found in "Three Stage Variable-Shift Iterations for the
// Solution of Polynomial Equations With a Posteriori Error Bounds for the
// Zeros" by M.A. Jenkins, Doctoral Thesis, Stanford Univeristy, 1969.
//
// NOTE: we assume the leading term of quadratic_sigma is 1.0.
VectorXd JenkinsTraubSolver::ComputeNextSigma() {
  const double u = sigma_(1);
  const double v = sigma_(2);

  const VectorXd::ReverseReturnType& creverse_k_polynomial =
      k_polynomial_.reverse();
  const VectorXd::ReverseReturnType& creverse_polynomial =
      polynomial_.reverse();

  const double b1 = -creverse_k_polynomial(0) / creverse_polynomial(0);
  const double b2 = -(creverse_k_polynomial(1) + b1 * creverse_polynomial(1)) /
                    creverse_polynomial(0);

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
  VectorXd new_quadratic_sigma(3);
  new_quadratic_sigma(0) = 1.0;
  new_quadratic_sigma(1) = u + delta_u;
  new_quadratic_sigma(2) = v + delta_v;
  return new_quadratic_sigma;
}

}  // namespace

bool FindPolynomialRootsJenkinsTraub(const VectorXd& polynomial,
                                     VectorXd* real_roots,
                                     VectorXd* complex_roots) {
  JenkinsTraubSolver solver(polynomial, real_roots, complex_roots);
  return solver.ExtractRoots();
}

}  // namespace rpoly_plus_plus
