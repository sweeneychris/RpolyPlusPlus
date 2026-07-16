// Self-contained Aberth-Ehrlich simultaneous root finder, used as an
// independent benchmark baseline (a different algorithm class than the
// deflation-based Jenkins-Traub solver under test).
//
// This is an original implementation of the public-domain algorithm -- it
// vendors no third-party code and carries no external dependency beyond Eigen
// (only for the VectorXd input type). It is intentionally simple; it is a
// reference point for accuracy/speed comparison, not a production solver.
//
// Initialization follows Bini's Newton-polygon construction (the same idea
// MPSolve uses): root moduli are estimated from the upper convex hull of
// {(i, log|a_i|)}, which places the initial guesses at the right magnitudes
// even for polynomials whose roots span many orders of magnitude.
#ifndef RPOLY_BENCHMARK_SOLVERS_ABERTH_H_
#define RPOLY_BENCHMARK_SOLVERS_ABERTH_H_

#include <cmath>
#include <complex>
#include <vector>

#include <Eigen/Core>

namespace rpoly_bench {

// Solves the real polynomial `poly` (highest-degree coefficient first, matching
// FindPolynomialRootsJenkinsTraub). Sets *ok to whether the iteration
// converged; the roots are returned either way so callers can measure residual.
inline std::vector<std::complex<double>> AberthSolve(const Eigen::VectorXd& poly,
                                                     bool* ok) {
  using Cx = std::complex<double>;
  *ok = false;
  std::vector<Cx> roots;

  int n = static_cast<int>(poly.size()) - 1;
  if (n < 1) {
    *ok = true;
    return roots;
  }

  // Ascending coefficients a[k] = coeff of x^k (a[n] is the leading term).
  std::vector<double> a(n + 1);
  for (int k = 0; k <= n; ++k) a[k] = poly(n - k);

  // Peel off exact zero roots (a[0]==0 means x divides p).
  int zero_roots = 0;
  while (zero_roots < n && a[zero_roots] == 0.0) ++zero_roots;
  for (int z = 0; z < zero_roots; ++z) roots.emplace_back(0.0, 0.0);
  const int lo = zero_roots;              // first nonzero ascending index
  const int d = n - zero_roots;           // degree of the deflated polynomial
  if (d == 0) {
    *ok = true;
    return roots;
  }

  // --- Newton-polygon initialization (upper convex hull of (k, log|a_k|)) ---
  std::vector<int> idx;                    // indices with nonzero coeff, k in [0,d]
  std::vector<double> ax(d + 1), ay(d + 1);
  for (int k = 0; k <= d; ++k) {
    const double c = a[lo + k];
    ax[k] = static_cast<double>(k);
    ay[k] = (c != 0.0) ? std::log(std::abs(c)) : 0.0;
    if (c != 0.0) idx.push_back(k);
  }
  // Andrew's monotone chain, upper hull: pop while the turn is not clockwise.
  std::vector<int> hull;
  for (int t = 0; t < static_cast<int>(idx.size()); ++t) {
    const int k = idx[t];
    while (hull.size() >= 2) {
      const int o = hull[hull.size() - 2], b = hull[hull.size() - 1];
      const double cross = (ax[b] - ax[o]) * (ay[k] - ay[o]) -
                           (ay[b] - ay[o]) * (ax[k] - ax[o]);
      if (cross >= 0.0)
        hull.pop_back();
      else
        break;
    }
    hull.push_back(k);
  }

  std::vector<Cx> z;
  z.reserve(d);
  const double kOffset = 0.517;            // avoid landing on the real axis
  double seg_phase = 0.0;
  for (std::size_t s = 0; s + 1 < hull.size(); ++s) {
    const int ki = hull[s], kj = hull[s + 1];
    const int cnt = kj - ki;
    const double modulus =
        std::pow(std::abs(a[lo + ki]) / std::abs(a[lo + kj]),
                 1.0 / static_cast<double>(cnt));
    for (int t = 0; t < cnt; ++t) {
      const double ang =
          2.0 * M_PI * t / cnt + kOffset + seg_phase;
      z.emplace_back(modulus * std::cos(ang), modulus * std::sin(ang));
    }
    seg_phase += 0.7;                       // decorrelate circles
  }
  // Safety: if the hull produced the wrong count, pad on the unit circle.
  while (static_cast<int>(z.size()) < d) {
    const double ang = 2.0 * M_PI * z.size() / d + kOffset;
    z.emplace_back(std::cos(ang), std::sin(ang));
  }

  // Horner evaluation of the deflated polynomial and its derivative.
  auto eval = [&](const Cx& x, Cx* p, Cx* dp) {
    *p = a[n];                              // leading coeff (== a[lo + d])
    *dp = Cx(0.0, 0.0);
    for (int k = d - 1; k >= 0; --k) {
      *dp = *dp * x + *p;
      *p = *p * x + a[lo + k];
    }
  };

  // --- Aberth-Ehrlich iteration (converged roots are frozen, as in MPSolve:
  // once a correction is at the rounding floor, updating it further only adds
  // last-digit chatter and keeps the whole sweep from reporting convergence).
  const int kMaxIters = 500;
  std::vector<char> done(d, 0);
  int num_done = 0;
  for (int iter = 0; iter < kMaxIters && num_done < d; ++iter) {
    for (int i = 0; i < d; ++i) {
      if (done[i]) continue;
      Cx p, dp;
      eval(z[i], &p, &dp);
      if (dp == Cx(0.0, 0.0)) {
        done[i] = 1;
        ++num_done;
        continue;
      }
      const Cx ratio = p / dp;             // Newton correction
      Cx sum(0.0, 0.0);
      for (int j = 0; j < d; ++j) {
        if (j == i) continue;
        const Cx diff = z[i] - z[j];
        if (std::abs(diff) > 0.0) sum += 1.0 / diff;
      }
      const Cx denom = 1.0 - ratio * sum;
      if (denom == Cx(0.0, 0.0)) continue;
      const Cx w = ratio / denom;
      z[i] -= w;
      if (std::abs(w) <= 1e-14 * (1.0 + std::abs(z[i]))) {
        done[i] = 1;
        ++num_done;
      }
    }
  }

  for (const Cx& r : z) roots.push_back(r);
  *ok = (num_done == d);
  return roots;
}

}  // namespace rpoly_bench

#endif  // RPOLY_BENCHMARK_SOLVERS_ABERTH_H_
