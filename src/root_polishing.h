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

#ifndef RPOLY_PLUS_PLUS_ROOT_POLISHING_H_
#define RPOLY_PLUS_PLUS_ROOT_POLISHING_H_

// Internal: Newton polishing of a computed root against the original
// polynomial. Exposed in a header so the step can be unit-tested directly;
// not part of the public API.

#include <Eigen/Core>

#include <cmath>
#include <complex>
#include <limits>

namespace rpoly_plus_plus {
namespace internal {

// True if a Horner-evaluated derivative p'(x) is indistinguishable from
// rounding noise, i.e. effectively zero. Newton polishing must stop then: at a
// multiple root p and p' are both noise, so the step p/p' is arbitrary and can
// even land on a *different* root, whose tiny residual looks like progress.
//
// No fixed threshold works here, because the size of p' scales with the
// coefficients and with |x|. Instead we compare against the rounding error of
// the evaluation itself. `abs_dp` is q'(|x|), where q(t) = sum |a_i| t^i is
// the absolute-value polynomial, computed by the same nested Horner pass. The
// forward error of the computed p'(x) is bounded by gamma_{2n} * q'(|x|) ~=
// 2n * eps * q'(|x|) (cf. Higham, "Accuracy and Stability of Numerical
// Algorithms", 2nd ed., sec. 5.1). We use 4n * eps to also cover the larger
// rounding constant of complex multiply-add. The bound scales with the
// coefficients and with |x|, so it is valid for any polynomial.
inline bool DerivativeIsNoise(const double abs_dp_computed, const double abs_dp,
                              const Eigen::Index n) {
  constexpr double kEps = std::numeric_limits<double>::epsilon();
  return abs_dp_computed <= 4.0 * static_cast<double>(n) * kEps * abs_dp;
}

// |x|. For complex x, sqrt(norm(x)) is several times cheaper than std::abs
// (hypot), which guards against overflow; fall back to it only when the
// squared magnitude overflows or loses precision to underflow.
inline double Magnitude(const double x) { return std::abs(x); }
inline double Magnitude(const std::complex<double>& x) {
  const double n2 = std::norm(x);
  if (n2 >= std::numeric_limits<double>::min() &&
      n2 <= std::numeric_limits<double>::max()) {
    return std::sqrt(n2);
  }
  return std::abs(x);
}

// Up to two Newton steps on `polynomial` (highest-degree coefficient first)
// starting from `x`. T is double (real roots: ~4x cheaper than a complex pass,
// and the root stays exactly real) or std::complex<double>. A step is taken
// only if p'(x) is above its rounding noise and the residual strictly
// decreases; otherwise `x` is returned unchanged from that point.
template <typename T, typename Derived>
T PolishRoot(const Eigen::MatrixBase<Derived>& polynomial, T x) {
  const Eigen::Index n = polynomial.size();
  for (int it = 0; it < 2; ++it) {
    T p = 0.0, dp = 0.0;
    double abs_p = 0.0, abs_dp = 0.0;
    const double abs_x = Magnitude(x);
    for (Eigen::Index i = 0; i < n; ++i) {
      dp = dp * x + p;
      p = p * x + polynomial(i);
      abs_dp = abs_dp * abs_x + abs_p;
      abs_p = abs_p * abs_x + std::abs(polynomial(i));
    }
    if (DerivativeIsNoise(Magnitude(dp), abs_dp, n)) break;
    const T x_new = x - p / dp;
    T p_new = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
      p_new = p_new * x_new + polynomial(i);
    }
    if (!(Magnitude(p_new) < Magnitude(p))) break;
    x = x_new;
  }
  return x;
}

}  // namespace internal
}  // namespace rpoly_plus_plus

#endif  // RPOLY_PLUS_PLUS_ROOT_POLISHING_H_
