// Adapter for MPSolve (Bini & Robol) -- the multiprecision Aberth-Ehrlich
// solver used here as a high-accuracy ORACLE for ground-truth root positions.
// https://github.com/robol/MPSolve
//
// This header vendors NO third-party code. It is compiled only when the build
// is configured with -DWITH_MPSOLVE=ON, which locates libmps + GMP and links
// them (see third_party/mpsolve/README.md). MPSolve is asked for high-precision
// approximations, then rounded to double for comparison against the double
// solvers -- so its roots serve as the reference "truth" for families that have
// no analytic closed form.
#ifndef RPOLY_BENCHMARK_SOLVERS_MPSOLVE_ADAPTER_H_
#define RPOLY_BENCHMARK_SOLVERS_MPSOLVE_ADAPTER_H_

#include <complex>
#include <cstdlib>
#include <vector>

#include <Eigen/Core>

#include <mps/mps.h>

namespace rpoly_bench {

// Solves `poly` (highest-degree coefficient first) via MPSolve at high internal
// precision and returns the roots rounded to double. Sets *ok on success.
inline std::vector<std::complex<double>> MpsolveSolve(const Eigen::VectorXd& poly,
                                                      bool* ok) {
  using Cx = std::complex<double>;
  *ok = false;
  std::vector<Cx> roots;

  const int n = static_cast<int>(poly.size()) - 1;
  if (n < 1) return roots;

  mps_context* ctx = mps_context_new();
  mps_monomial_poly* mp = mps_monomial_poly_new(ctx, n);
  // MPSolve indexes coefficients by power: index i is the coeff of x^i, so the
  // constant term poly(n) goes to index 0 and the leading poly(0) to index n.
  for (int i = 0; i <= n; ++i) {
    mps_monomial_poly_set_coefficient_d(ctx, mp, i, poly(n - i), 0.0);
  }
  mps_context_set_input_poly(ctx, MPS_POLYNOMIAL(mp));
  mps_context_set_output_prec(ctx, 200);  // internal precision, bits
  mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE);

  mps_mpsolve(ctx);

  cplx_t* results = NULL;
  mps_context_get_roots_d(ctx, &results, NULL);
  if (results != NULL) {
    for (int i = 0; i < n; ++i) {
      roots.emplace_back(cplx_Re(results[i]), cplx_Im(results[i]));
    }
    free(results);
    *ok = true;
  }

  mps_context_free(ctx);
  return roots;
}

}  // namespace rpoly_bench

#endif  // RPOLY_BENCHMARK_SOLVERS_MPSOLVE_ADAPTER_H_
