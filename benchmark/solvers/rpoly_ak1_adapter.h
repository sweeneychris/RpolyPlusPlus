// Adapter for David Binner's RPOLY_AK1 -- a self-contained C++ translation of
// Jenkins & Traub's original RPOLY (ACM TOMS Algorithm 493), available at
// https://www.akiti.ca/rpoly_ak1_cpp.html
//
// This header vendors NO third-party code. It is compiled only when the build
// is configured with -DWITH_RPOLY=ON, which also compiles the source file you
// place at third_party/rpoly/rpoly_ak1.cpp (see third_party/rpoly/README.md).
//
// RPOLY_AK1 exposes (highest-degree coefficient first):
//
//     void rpoly_ak1(double op[MDP1], int* Degree,
//                    double zeror[MAXDEGREE], double zeroi[MAXDEGREE]);
//
// with MAXDEGREE == 100. On entry *Degree is the polynomial degree; on exit it
// is the number of zeros found, or a value < the input degree on failure. If
// the revision you download differs, adjust only the declaration and the
// success test below.
#ifndef RPOLY_BENCHMARK_SOLVERS_RPOLY_AK1_ADAPTER_H_
#define RPOLY_BENCHMARK_SOLVERS_RPOLY_AK1_ADAPTER_H_

#include <complex>
#include <vector>

#include <Eigen/Core>

// Provided by third_party/rpoly/rpoly_ak1.cpp (its demo main() removed/renamed).
void rpoly_ak1(double op[], int* Degree, double zeror[], double zeroi[]);

namespace rpoly_bench {

inline std::vector<std::complex<double>> RpolyAk1Solve(
    const Eigen::VectorXd& poly, bool* ok) {
  using Cx = std::complex<double>;
  *ok = false;
  std::vector<Cx> roots;

  const int n = static_cast<int>(poly.size()) - 1;
  if (n < 1 || n > 100) return roots;  // RPOLY_AK1 caps at MAXDEGREE = 100.

  double op[101];
  for (int i = 0; i <= n; ++i) op[i] = poly(i);  // highest-degree first
  int degree = n;
  double zeror[100] = {0.0};
  double zeroi[100] = {0.0};

  rpoly_ak1(op, &degree, zeror, zeroi);

  if (degree < 1) return roots;  // AK1 signals failure with Degree < 1.
  for (int i = 0; i < degree; ++i) roots.emplace_back(zeror[i], zeroi[i]);
  *ok = (static_cast<int>(roots.size()) == n);
  return roots;
}

}  // namespace rpoly_bench

#endif  // RPOLY_BENCHMARK_SOLVERS_RPOLY_AK1_ADAPTER_H_
