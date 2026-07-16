// Comprehensive accuracy / speed evaluation for FindPolynomialRootsJenkinsTraub.
//
// Three pillars:
//   1. Literature-standard numerical stress families (Wilkinson, Mignotte,
//      Kac, cyclotomic, multiplicity ladders, clustered roots) with
//      CONDITION-AWARE pass gates: thresholds are set at what a backward-
//      stable solver can achieve for each family, so a pass means
//      "condition-limited accuracy", the strongest claim any solver can make.
//   2. Computer-vision / geometric-solver polynomial families generated from
//      synthetic geometry with verifiable ground truth:
//        * P3P (Grunert) quartics       -- absolute pose from 3 points
//        * 7-point fundamental cubics   -- det(F1 + lambda*F2) = 0
//        * ray-torus quartics           -- incl. grazing (tangent) rays
//        * point-to-ellipse quartics    -- closest-point distance
//   3. A companion-matrix eigenvalue oracle (the "textbook robust" method)
//      for differential accuracy checking and speed context.
//
// Every family has an explicit PASS/FAIL gate; the process exit code is the
// number of failed gates. An optional CSV corpus of real coefficient rows
// (leading coefficient first) can be replayed with:  rpoly_benchmark corpus.csv
//
// Heap-free probe (optional, portable): the rpoly_benchmark_heapcheck target
// compiles this file together with the solver sources and defines
// RPOLY_BENCH_NO_MALLOC_GATE, which routes every Eigen allocation through a
// throwing gate (benchmark/eigen_malloc_gate.h). A single solve that allocates
// then trips the gate and is reported as "no"; a stack-resident solve reports
// "yes". The plain rpoly_benchmark target links the shared library and reports
// "n/a". Unlike glibc malloc interposition, this works on macOS and Linux.

// The gate must be established before any Eigen header is included. For the
// heapcheck target this same header is also force-included into the solver
// sources by the build system, so the solver's allocations are observed too.
#ifdef RPOLY_BENCH_NO_MALLOC_GATE
#include "benchmark/eigen_malloc_gate.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "src/find_polynomial_roots_jenkins_traub.h"

// Optional baseline / oracle solvers. Aberth-Ehrlich is always available
// (self-contained); RPOLY and MPSolve are compiled in only when their sources
// are present (see third_party/*/README.md).
#include "benchmark/solvers/aberth.h"
#ifdef WITH_RPOLY
#include "benchmark/solvers/rpoly_ak1_adapter.h"
#endif
#ifdef WITH_MPSOLVE
#include "benchmark/solvers/mpsolve_adapter.h"
#endif

namespace {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using rpoly_plus_plus::FindPolynomialRootsJenkinsTraub;
using Complex = std::complex<double>;
using Clock = std::chrono::steady_clock;
constexpr double kEps = std::numeric_limits<double>::epsilon();

// ------------------------------ utilities ---------------------------------

VectorXd PolyFromRoots(const std::vector<Complex>& roots) {
  std::vector<Complex> c{Complex(1.0, 0.0)};
  for (const Complex& r : roots) {
    std::vector<Complex> next(c.size() + 1, Complex(0.0, 0.0));
    for (std::size_t i = 0; i < c.size(); ++i) {
      next[i] += c[i];
      next[i + 1] -= r * c[i];
    }
    c.swap(next);
  }
  VectorXd poly(static_cast<int>(c.size()));
  for (std::size_t i = 0; i < c.size(); ++i)
    poly(static_cast<int>(i)) = c[i].real();
  return poly;
}

// Normalized backward residual |P(z)| / sum_i |a_i||z|^{n-i}. For a
// backward-stable solver this is O(n * eps) for every returned root,
// regardless of conditioning -- it is the universal correctness certificate.
double NormalizedResidual(const VectorXd& poly, const Complex& z) {
  Complex p(0.0, 0.0);
  double bound = 0.0;
  const double az = std::abs(z);
  for (int i = 0; i < poly.size(); ++i) {
    p = p * z + poly(i);
    bound = bound * az + std::abs(poly(i));
  }
  return bound > 0.0 ? std::abs(p) / bound : std::abs(p);
}

// Greedy nearest-neighbour matching; RELATIVE forward error per root
// (|dz| / max(|true|, floor)) so wide-dynamic-range roots are scored fairly.
double MaxRelForwardError(std::vector<Complex> computed,
                          const std::vector<Complex>& truth,
                          double abs_floor = 1e-12) {
  double worst = 0.0;
  for (const Complex& t : truth) {
    if (computed.empty()) return std::numeric_limits<double>::infinity();
    std::size_t best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < computed.size(); ++j) {
      const double d = std::abs(computed[j] - t);
      if (d < best_d) {
        best_d = d;
        best = j;
      }
    }
    worst = std::max(worst, best_d / std::max(std::abs(t), abs_floor));
    computed.erase(computed.begin() + static_cast<long>(best));
  }
  return worst;
}

std::vector<Complex> Solve(const VectorXd& poly, bool* ok) {
  VectorXd re, im;
  *ok = FindPolynomialRootsJenkinsTraub(poly, &re, &im);
  std::vector<Complex> out;
  if (!*ok) return out;
  for (int i = 0; i < re.size(); ++i) out.emplace_back(re(i), im(i));
  return out;
}

// Companion-matrix eigenvalue oracle (balanced QR via Eigen).
std::vector<Complex> CompanionRoots(const VectorXd& p) {
  const int n = static_cast<int>(p.size()) - 1;
  MatrixXd C = MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) C(i, n - 1) = -p(n - i) / p(0);
  for (int i = 1; i < n; ++i) C(i, i - 1) = 1.0;
  Eigen::EigenSolver<MatrixXd> es(C);
  std::vector<Complex> r;
  for (int i = 0; i < n; ++i) r.push_back(es.eigenvalues()(i));
  return r;
}

// ------------------------------ gate ledger --------------------------------

struct Gate {
  std::string name;
  double measured;
  double threshold;
  bool pass;
  std::string note;
};
std::vector<Gate>& Gates() {
  static std::vector<Gate> g;
  return g;
}
bool Check(const std::string& name, double measured, double threshold,
           const std::string& note = "") {
  const bool pass = measured <= threshold;
  Gates().push_back({name, measured, threshold, pass, note});
  std::printf("  %-46s %11.3e <= %9.1e  [%s]%s%s\n", name.c_str(), measured,
              threshold, pass ? "PASS" : "FAIL", note.empty() ? "" : "  # ",
              note.c_str());
  return pass;
}

// ============================================================================
// Pillar 1: literature-standard numerical families
// ============================================================================

void RunClassicFamilies() {
  std::printf("\n== Classic numerical stress families ==\n");
  std::mt19937 g(12345);
  std::uniform_real_distribution<double> u(-5.0, 5.0);

  // ---- well-separated real roots: near machine precision expected ----
  {
    double fwd = 0, res = 0, fails = 0;
    for (int k = 0; k < 100; ++k) {
      std::vector<double> r;
      while (static_cast<int>(r.size()) < 10) {
        const double x = u(g);
        bool ok = true;
        for (double y : r)
          if (std::abs(x - y) < 0.25) ok = false;
        if (ok) r.push_back(x);
      }
      std::vector<Complex> roots;
      for (double x : r) roots.emplace_back(x, 0.0);
      const VectorXd p = PolyFromRoots(roots);
      bool ok;
      const auto got = Solve(p, &ok);
      if (!ok) {
        fails++;
        continue;
      }
      fwd = std::max(fwd, MaxRelForwardError(got, roots));
      for (const auto& z : got)
        res = std::max(res, NormalizedResidual(p, z));
    }
    Check("separated real deg-10: solver failures", fails, 0);
    Check("separated real deg-10: rel fwd error", fwd, 1e-8);
    Check("separated real deg-10: residual", res, 1e-13);
  }

  // ---- clustered roots: resolution-limited by conditioning ----
  // m roots within radius delta of 1.0 can only be resolved to
  // ~ eps^{1/m}; a tighter gate would fail EVERY backward-stable solver.
  {
    std::normal_distribution<double> nd(0.0, 1e-3);
    double fwd = 0, res = 0, fails = 0;
    for (int k = 0; k < 50; ++k) {
      std::vector<Complex> roots;
      for (int i = 0; i < 8; ++i) roots.emplace_back(1.0 + nd(g), 0.0);
      const VectorXd p = PolyFromRoots(roots);
      bool ok;
      const auto got = Solve(p, &ok);
      if (!ok) {
        fails++;
        continue;
      }
      fwd = std::max(fwd, MaxRelForwardError(got, roots));
      for (const auto& z : got)
        res = std::max(res, NormalizedResidual(p, z));
    }
    const double cond_limit = std::pow(kEps, 1.0 / 8.0);  // ~ 1.1e-2
    Check("clustered 8@1e-3: solver failures", fails, 0);
    Check("clustered 8@1e-3: rel fwd error", fwd, 10.0 * cond_limit,
          "resolution-limited: eps^{1/8} ~ 1e-2");
    Check("clustered 8@1e-3: residual", res, 1e-12);
  }

  // ---- complex conjugate pairs ----
  {
    std::uniform_real_distribution<double> u3(-3.0, 3.0);
    double fwd = 0, res = 0, fails = 0;
    for (int k = 0; k < 100; ++k) {
      std::vector<Complex> roots;
      for (int i = 0; i < 5; ++i) {
        const double re = u3(g), im = std::abs(u3(g)) + 0.1;
        roots.emplace_back(re, im);
        roots.emplace_back(re, -im);
      }
      const VectorXd p = PolyFromRoots(roots);
      bool ok;
      const auto got = Solve(p, &ok);
      if (!ok) {
        fails++;
        continue;
      }
      fwd = std::max(fwd, MaxRelForwardError(got, roots));
      for (const auto& z : got)
        res = std::max(res, NormalizedResidual(p, z));
    }
    Check("complex pairs deg-10: solver failures", fails, 0);
    Check("complex pairs deg-10: rel fwd error", fwd, 1e-8);
    Check("complex pairs deg-10: residual", res, 1e-13);
  }

  // ---- wide dynamic range 1e-4 .. 1e4 (RELATIVE error is the fair metric)
  {
    std::vector<Complex> roots;
    for (int k = -4; k <= 4; ++k) roots.emplace_back(std::pow(10.0, k), 0.0);
    const VectorXd p = PolyFromRoots(roots);
    bool ok;
    const auto got = Solve(p, &ok);
    double fwd = ok ? MaxRelForwardError(got, roots)
                    : std::numeric_limits<double>::infinity();
    double res = 0;
    for (const auto& z : got) res = std::max(res, NormalizedResidual(p, z));
    Check("wide range 1e-4..1e4: rel fwd error", fwd, 1e-9);
    Check("wide range 1e-4..1e4: residual", res, 1e-14);
  }

  // ---- Wilkinson 10 / 15: severe coefficient-space conditioning ----
  for (const int n : {10, 15}) {
    std::vector<Complex> roots;
    for (int i = 1; i <= n; ++i) roots.emplace_back(i, 0.0);
    const VectorXd p = PolyFromRoots(roots);
    bool ok;
    const auto got = Solve(p, &ok);
    double fwd = ok ? MaxRelForwardError(got, roots)
                    : std::numeric_limits<double>::infinity();
    double res = 0;
    for (const auto& z : got) res = std::max(res, NormalizedResidual(p, z));
    char nm[64];
    std::snprintf(nm, sizeof(nm), "wilkinson-%d: rel fwd error", n);
    // Middle roots of W15 have kappa ~ 1e12; eps*kappa is the honest limit.
    Check(nm, fwd, n == 10 ? 1e-8 : 1e-3, "condition-limited");
    std::snprintf(nm, sizeof(nm), "wilkinson-%d: residual", n);
    Check(nm, res, 1e-14);
  }

  // ---- multiplicity ladder (x-1)^m (x+2): fwd limited to eps^{1/m} ----
  for (const int m : {2, 3, 4, 5}) {
    std::vector<Complex> roots(static_cast<size_t>(m), Complex(1, 0));
    roots.emplace_back(-2.0, 0.0);
    const VectorXd p = PolyFromRoots(roots);
    bool ok;
    const auto got = Solve(p, &ok);
    double fwd = ok ? MaxRelForwardError(got, roots)
                    : std::numeric_limits<double>::infinity();
    double res = 0;
    for (const auto& z : got) res = std::max(res, NormalizedResidual(p, z));
    char nm[64];
    std::snprintf(nm, sizeof(nm), "(x-1)^%d (x+2): rel fwd error", m);
    Check(nm, fwd, 100.0 * std::pow(kEps, 1.0 / m), "eps^{1/m} limit");
    std::snprintf(nm, sizeof(nm), "(x-1)^%d (x+2): residual", m);
    Check(nm, res, 1e-13);
  }

  // ---- RPOLY documentation case: triple root + simple roots ----
  {
    std::vector<Complex> roots = {{0.1, 0}, {0.1, 0}, {0.1, 0},
                                  {0.5, 0}, {0.6, 0}, {0.7, 0}};
    const VectorXd p = PolyFromRoots(roots);
    bool ok;
    const auto got = Solve(p, &ok);
    double fwd = ok ? MaxRelForwardError(got, roots)
                    : std::numeric_limits<double>::infinity();
    Check("rpoly triple-0.1 case: rel fwd error", fwd,
          100.0 * std::pow(kEps, 1.0 / 3.0), "triple root limit");
  }

  // ---- cyclotomic x^n - 1: all roots on unit circle (issue-#8 regime) ----
  {
    double fwd = 0, fails = 0;
    for (int n = 2; n <= 24; ++n) {
      VectorXd p = VectorXd::Zero(n + 1);
      p(0) = 1;
      p(n) = -1;
      std::vector<Complex> roots;
      for (int k = 0; k < n; ++k)
        roots.emplace_back(std::cos(2 * M_PI * k / n),
                           std::sin(2 * M_PI * k / n));
      bool ok;
      const auto got = Solve(p, &ok);
      if (!ok) {
        fails++;
        continue;
      }
      fwd = std::max(fwd, MaxRelForwardError(got, roots));
    }
    Check("cyclotomic x^n-1, n=2..24: failures", fails, 0);
    Check("cyclotomic x^n-1, n=2..24: rel fwd error", fwd, 1e-10);
  }

  // ---- Mignotte x^9 - 2(ax-1)^2: canonical near-double cluster ----
  for (const double a : {10.0, 100.0}) {
    const int n = 9;
    VectorXd p = VectorXd::Zero(n + 1);
    p(0) = 1;
    p(n - 2) += -2 * a * a;
    p(n - 1) += 4 * a;
    p(n) += -2;
    bool ok;
    const auto got = Solve(p, &ok);
    double res = ok ? 0.0 : std::numeric_limits<double>::infinity();
    for (const auto& z : got) res = std::max(res, NormalizedResidual(p, z));
    char nm[64];
    std::snprintf(nm, sizeof(nm), "mignotte(9, a=%g): residual", a);
    Check(nm, res, 1e-12);
  }

  // ---- Kac polynomials: random coefficients, roots crowd the unit circle
  {
    std::normal_distribution<double> nd(0.0, 1.0);
    double res = 0, fails = 0;
    for (int deg : {8, 14, 20}) {
      for (int k = 0; k < 500; ++k) {
        VectorXd p(deg + 1);
        for (int i = 0; i <= deg; ++i) p(i) = nd(g);
        if (std::abs(p(0)) < 1e-3) p(0) = 1.0;
        bool ok;
        const auto got = Solve(p, &ok);
        if (!ok) {
          fails++;
          continue;
        }
        for (const auto& z : got)
          res = std::max(res, NormalizedResidual(p, z));
      }
    }
    Check("kac deg 8/14/20 (1500): failures", fails, 0);
    Check("kac deg 8/14/20 (1500): residual", res, 1e-11);
  }
}

// ============================================================================
// Pillar 2: computer-vision / geometric polynomial families
// ============================================================================

// Small dense polynomial helpers over v (coefficients ascending in these
// helpers for clarity; converted to solver order at the end).
using Poly = std::vector<double>;  // p[i] * v^i
Poly Mul(const Poly& a, const Poly& b) {
  Poly r(a.size() + b.size() - 1, 0.0);
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < b.size(); ++j) r[i + j] += a[i] * b[j];
  return r;
}
Poly Add(Poly a, const Poly& b, double s = 1.0) {
  if (b.size() > a.size()) a.resize(b.size(), 0.0);
  for (size_t i = 0; i < b.size(); ++i) a[i] += s * b[i];
  return a;
}
VectorXd ToSolverOrder(const Poly& p) {
  VectorXd out(static_cast<int>(p.size()));
  for (size_t i = 0; i < p.size(); ++i)
    out(static_cast<int>(i)) = p[p.size() - 1 - i];
  return out;
}
double EvalAsc(const Poly& p, double v) {
  double r = 0;
  for (size_t i = p.size(); i-- > 0;) r = r * v + p[i];
  return r;
}

// ---- P3P (Grunert): quartic in v = s3/s1 from three 3D points + camera ----
// Ground truth: the geometric depth ratio must appear among the real roots.
void RunP3P() {
  std::printf("\n== P3P (Grunert) quartics from synthetic camera geometry ==\n");
  std::mt19937 g(777);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  double worst_truth_err = 0, worst_res = 0;
  int fails = 0, instances = 0;
  while (instances < 300) {
    // Camera at origin; three points in front of the camera.
    Vector3d P1(u(g), u(g), 4.0 + u(g));
    Vector3d P2(u(g), u(g), 4.0 + u(g));
    Vector3d P3(u(g), u(g), 4.0 + u(g));
    const double s1 = P1.norm(), s2 = P2.norm(), s3 = P3.norm();
    const double a = (P2 - P3).norm(), b = (P1 - P3).norm(),
                 c = (P1 - P2).norm();
    if (a < 0.3 || b < 0.3 || c < 0.3) continue;  // avoid degenerate triangles
    const double ca = P2.dot(P3) / (s2 * s3);   // cos(alpha)
    const double cb = P1.dot(P3) / (s1 * s3);   // cos(beta)
    const double cg = P1.dot(P2) / (s1 * s2);   // cos(gamma)
    const double A = (a * a) / (b * b), C = (c * c) / (b * b);
    // Elimination (see derivation in repo notes):
    //   u*(2*cg - 2*ca*v) = (A - C)*(1 + v^2 - 2*cb*v) + 1 - v^2 =: N(v)
    //   N^2 - 2*cg*N*D + D^2*(1 - C*(1 + v^2 - 2*cb*v)) = 0, D = 2*cg - 2*ca*v
    const Poly W = {1.0, -2.0 * cb, 1.0};                 // 1 - 2 cb v + v^2
    Poly N = {1.0, 0.0, -1.0};                            // 1 - v^2
    N = Add(N, W, A - C);
    const Poly D = {2.0 * cg, -2.0 * ca};
    if (std::abs(EvalAsc(D, s3 / s1)) < 1e-3) continue;   // avoid pole at truth
    Poly quartic = Mul(N, N);
    quartic = Add(quartic, Mul(N, D), -2.0 * cg);
    Poly tail = Add(Poly{1.0}, W, -C);                    // 1 - C*W(v)
    quartic = Add(quartic, Mul(Mul(D, D), tail));
    if (quartic.size() != 5 || std::abs(quartic[4]) < 1e-8) continue;
    const double v_true = s3 / s1;
    // Construction sanity: truth must satisfy the quartic.
    if (std::abs(EvalAsc(quartic, v_true)) > 1e-9 *
            (std::abs(quartic[0]) + std::abs(quartic[1]) +
             std::abs(quartic[2]) + std::abs(quartic[3]) +
             std::abs(quartic[4]))) {
      continue;  // should not happen; skip rather than mis-grade the solver
    }
    ++instances;
    const VectorXd p = ToSolverOrder(quartic);
    bool ok;
    const auto got = Solve(p, &ok);
    if (!ok) {
      ++fails;
      continue;
    }
    double best = std::numeric_limits<double>::infinity();
    for (const auto& z : got) {
      if (std::abs(z.imag()) < 1e-6)
        best = std::min(best, std::abs(z.real() - v_true) / v_true);
      worst_res = std::max(worst_res, NormalizedResidual(p, z));
    }
    worst_truth_err = std::max(worst_truth_err, best);
  }
  Check("p3p grunert (300): solver failures", fails, 0);
  Check("p3p grunert (300): depth-ratio recovery err", worst_truth_err, 1e-9);
  Check("p3p grunert (300): residual", worst_res, 1e-12);
}

// ---- 7-point fundamental matrix: det(F1 + lambda F2) cubic ----
void RunSevenPoint() {
  std::printf("\n== 7-point fundamental-matrix cubics ==\n");
  std::mt19937 g(4242);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  double worst_det = 0;
  int fails = 0, no_real = 0, instances = 0;
  while (instances < 300) {
    // Random relative pose (modest rotation) and 7 points.
    const Vector3d axis = Vector3d(u(g), u(g), u(g)).normalized();
    const double angle = 0.3 * u(g);
    const Eigen::AngleAxisd R(angle, axis);
    const Vector3d t = Vector3d(u(g), u(g), 0.2 * u(g)).normalized();
    MatrixXd A(7, 9);
    for (int i = 0; i < 7; ++i) {
      const Vector3d P(2.0 * u(g), 2.0 * u(g), 5.0 + 2.0 * u(g));
      const Vector3d Q = R * P + t;
      if (P.z() < 0.5 || Q.z() < 0.5) {
        A.setZero();
        break;
      }
      const double x = P.x() / P.z(), y = P.y() / P.z();
      const double xp = Q.x() / Q.z(), yp = Q.y() / Q.z();
      A.row(i) << xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1.0;
    }
    if (A.isZero(0)) continue;
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeFullV);
    if (svd.singularValues()(5) < 1e-10) continue;  // degenerate config
    Eigen::Matrix3d F1, F2;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        F1(r, c) = svd.matrixV()(3 * r + c, 7);
        F2(r, c) = svd.matrixV()(3 * r + c, 8);
      }
    // det(F1 + lambda F2) is cubic in lambda; interpolate at 4 nodes.
    auto f = [&](double l) { return (F1 + l * F2).determinant(); };
    const double f0 = f(0), f1 = f(1), fm1 = f(-1), f2 = f(2);
    const double c0 = f0;
    const double c2 = 0.5 * (f1 + fm1) - f0;
    const double s = f1 - fm1;
    const double c3 = (f2 - f0 - 4.0 * c2 - s) / 6.0;
    const double c1 = 0.5 * s - c3;
    VectorXd p(4);
    p << c3, c2, c1, c0;
    const double pmax = p.cwiseAbs().maxCoeff();
    if (pmax < 1e-14 || std::abs(c3) < 1e-10 * pmax) continue;
    p /= pmax;
    ++instances;
    bool ok;
    const auto got = Solve(p, &ok);
    if (!ok) {
      ++fails;
      continue;
    }
    bool any_real = false;
    for (const auto& z : got) {
      if (std::abs(z.imag()) > 1e-8) continue;
      any_real = true;
      const Eigen::Matrix3d F = F1 + z.real() * F2;
      const double nf = F.norm();
      worst_det = std::max(worst_det,
                           std::abs(F.determinant()) / (nf * nf * nf));
    }
    if (!any_real) ++no_real;  // a cubic always has a real root
  }
  Check("7-point (300): solver failures", fails, 0);
  Check("7-point (300): cubics w/o real root", no_real, 0,
        "odd degree must yield a real root");
  Check("7-point (300): normalized |det F|", worst_det, 1e-11);
}

// ---- ray-torus intersection quartics, incl. grazing rays (issue #5) ----
void RunTorus() {
  std::printf("\n== Ray-torus intersection quartics ==\n");
  std::mt19937 g(31337);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const double R = 1.0, r = 0.35;
  auto torus_quartic = [&](const Vector3d& o, const Vector3d& d) {
    // (|o+td|^2 + R^2 - r^2)^2 - 4 R^2 ((ox+t dx)^2 + (oy+t dy)^2)
    const Poly Q = {o.squaredNorm() + R * R - r * r, 2.0 * o.dot(d), 1.0};
    const Poly W = {o.x() * o.x() + o.y() * o.y(),
                    2.0 * (o.x() * d.x() + o.y() * d.y()),
                    d.x() * d.x() + d.y() * d.y()};
    return Add(Mul(Q, Q), W, -4.0 * R * R);
  };
  auto surface = [&](const Vector3d& x) {
    const double q = x.squaredNorm() + R * R - r * r;
    return q * q - 4.0 * R * R * (x.x() * x.x() + x.y() * x.y());
  };
  // (a) random hitting rays
  double worst_surface = 0, res = 0;
  int fails = 0, hits = 0;
  while (hits < 400) {
    const Vector3d o(3.0 * u(g), 3.0 * u(g), 1.5 * u(g));
    if (std::abs(std::sqrt(o.x() * o.x() + o.y() * o.y()) - R) < r + 0.2)
      continue;  // start outside the tube
    // aim at a random point of the tube's center circle
    const double phi = M_PI * u(g);
    const Vector3d target(R * std::cos(phi), R * std::sin(phi), 0.15 * u(g));
    const Vector3d d = (target - o).normalized();
    const VectorXd p = ToSolverOrder(torus_quartic(o, d));
    bool ok;
    const auto got = Solve(p, &ok);
    if (!ok) {
      ++fails;
      ++hits;
      continue;
    }
    bool real_hit = false;
    for (const auto& z : got) {
      res = std::max(res, NormalizedResidual(p, z));
      if (std::abs(z.imag()) < 1e-7 && z.real() > 0) {
        real_hit = true;
        const Vector3d x = o + z.real() * d;
        // scale-invariant surface residual
        worst_surface =
            std::max(worst_surface, std::abs(surface(x)) /
                                        (1.0 + x.squaredNorm() * x.squaredNorm()));
      }
    }
    if (real_hit) ++hits;
  }
  Check("torus random rays (400 hits): failures", fails, 0);
  Check("torus random rays: surface residual", worst_surface, 1e-9);
  Check("torus random rays: poly residual", res, 1e-12);

  // (b) grazing rays: tangent to the tube -> (near-)double roots
  double graze_res = 0;
  int graze_fail = 0;
  for (int k = 0; k < 200; ++k) {
    const double phi = M_PI * u(g), psi = M_PI * u(g);
    const Vector3d cdir(std::cos(phi), std::sin(phi), 0.0);
    const Vector3d n = std::cos(psi) * cdir + Vector3d(0, 0, std::sin(psi));
    const Vector3d s = R * cdir + r * n;      // point on the surface
    const Vector3d t1(-std::sin(phi), std::cos(phi), 0.0);  // tangent dir
    const Vector3d o = s - 5.0 * t1;
    const VectorXd p = ToSolverOrder(torus_quartic(o, t1));
    bool ok;
    const auto got = Solve(p, &ok);
    if (!ok) {
      ++graze_fail;
      continue;
    }
    for (const auto& z : got)
      graze_res = std::max(graze_res, NormalizedResidual(p, z));
  }
  Check("torus grazing rays (200): failures", graze_fail, 0,
        "tangency => double roots");
  Check("torus grazing rays: poly residual", graze_res, 1e-12);
}

// ---- point-to-ellipse closest distance quartics ----
void RunEllipse() {
  std::printf("\n== Point-to-ellipse distance quartics ==\n");
  std::mt19937 g(2718);
  std::uniform_real_distribution<double> u(0.05, 3.0);
  double worst_dist_err = 0;
  int fails = 0;
  for (int k = 0; k < 300; ++k) {
    const double A = 1.0 + u(g), B = 0.2 + 0.5 * u(g);  // A > B
    const double px = u(g), py = u(g);
    // f(t) = (A px)^2 (t+B^2)^2 + (B py)^2 (t+A^2)^2 - (t+A^2)^2 (t+B^2)^2
    const Poly ta = {A * A, 1.0}, tb = {B * B, 1.0};
    Poly q = Mul(Mul(ta, ta), Mul(tb, tb));
    for (auto& x : q) x = -x;
    q = Add(q, Mul(tb, tb), (A * px) * (A * px));
    q = Add(q, Mul(ta, ta), (B * py) * (B * py));
    const VectorXd p = ToSolverOrder(q);
    bool ok;
    const auto got = Solve(p, &ok);
    if (!ok) {
      ++fails;
      continue;
    }
    // Largest real root t* > -B^2 gives the closest point.
    double tstar = -std::numeric_limits<double>::infinity();
    for (const auto& z : got)
      if (std::abs(z.imag()) < 1e-8) tstar = std::max(tstar, z.real());
    const double cx = A * A * px / (tstar + A * A);
    const double cy = B * B * py / (tstar + B * B);
    const double dist =
        std::hypot(px - cx, py - cy);
    // Ground truth: coarse angular scan + golden-section refinement.
    auto dist_at = [&](double th) {
      return std::hypot(px - A * std::cos(th), py - B * std::sin(th));
    };
    double best_th = 0, best = std::numeric_limits<double>::infinity();
    const int M = 2000;
    for (int i = 0; i <= M; ++i) {
      const double th = 0.5 * M_PI * i / M;  // first quadrant (px,py>0)
      const double d0 = dist_at(th);
      if (d0 < best) { best = d0; best_th = th; }
    }
    double lo = best_th - M_PI / M, hi = best_th + M_PI / M;
    const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
    for (int it = 0; it < 80; ++it) {
      const double x1 = hi - gr * (hi - lo), x2 = lo + gr * (hi - lo);
      if (dist_at(x1) < dist_at(x2)) hi = x2; else lo = x1;
    }
    best = dist_at(0.5 * (lo + hi));
    worst_dist_err = std::max(worst_dist_err,
                              std::abs(dist - best) / std::max(best, 1e-9));
  }
  Check("ellipse distance (300): failures", fails, 0);
  Check("ellipse distance (300): dist vs brute force", worst_dist_err, 1e-6,
        "limited by brute-force sampling");
}

// ============================================================================
// Pillar 3: companion-matrix differential oracle + speed
// ============================================================================

void RunDifferentialAndSpeed() {
  std::printf("\n== Differential vs companion-matrix QR (accuracy) ==\n");
  std::mt19937 g(999);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  for (const int deg : {8, 16, 24}) {
    double worst = 0;
    int fails = 0, worse_than_oracle = 0;
    for (int k = 0; k < 50; ++k) {
      std::vector<Complex> roots;
      for (int d = 0; d < deg; ++d) roots.emplace_back(u(g), 0.0);
      const VectorXd p = PolyFromRoots(roots);
      bool ok;
      const auto got = Solve(p, &ok);
      if (!ok) {
        fails++;
        continue;
      }
      const auto oracle = CompanionRoots(p);
      // Differential claim: JT's BACKWARD error (normalized residual) is at
      // machine level, and no worse than the companion oracle's. Residuals
      // are condition-independent, so this cannot false-fail on clustered
      // random roots the way positional comparison does; positional
      // agreement is meaningless when both solvers sit in a large
      // conditioning ball, and companion QR's backward error lives in
      // matrix space, making it an unreliable positional oracle anyway.
      double r_jt = 0, r_c = 0;
      for (const auto& z : got) r_jt = std::max(r_jt, NormalizedResidual(p, z));
      for (const auto& z : oracle) r_c = std::max(r_c, NormalizedResidual(p, z));
      worst = std::max(worst, r_jt);
      if (r_jt > 10.0 * std::max(r_c, 1e-16)) ++worse_than_oracle;
    }
    char nm[64];
    std::snprintf(nm, sizeof(nm), "vs companion, deg %d: JT residual", deg);
    Check(nm, worst, 1e-13, "backward error at machine level");
    std::snprintf(nm, sizeof(nm), "vs companion, deg %d: JT >10x worse", deg);
    Check(nm, worse_than_oracle, 0, "instances where oracle beats JT 10x");
    std::snprintf(nm, sizeof(nm), "vs companion, deg %d: failures", deg);
    Check(nm, fails, 0);
  }

  std::printf("\n== Speed (mean over >=40ms of solves; heap probe) ==\n");
  std::printf("%8s %14s %16s %12s\n", "degree", "JT ns/solve",
              "companion ns", "heap-free");
  const int degrees[] = {4, 8, 16, 24, 32, 40, 64};
  for (const int deg : degrees) {
    std::vector<VectorXd> polys;
    for (int i = 0; i < 64; ++i) {
      std::vector<Complex> roots;
      for (int d = 0; d < deg; ++d) roots.emplace_back(u(g), 0.0);
      polys.push_back(PolyFromRoots(roots));
    }
    volatile double sink = 0.0;
    VectorXd re, im;
    for (const auto& p : polys)
      if (FindPolynomialRootsJenkinsTraub(p, &re, &im)) sink += re(0);
    const auto t0 = Clock::now();
    std::int64_t solves = 0;
    double el = 0;
    do {
      for (const auto& p : polys)
        if (FindPolynomialRootsJenkinsTraub(p, &re, &im)) sink += re(0);
      solves += static_cast<std::int64_t>(polys.size());
      el = std::chrono::duration<double>(Clock::now() - t0).count();
    } while (el < 0.04);
    const double jt_ns = el * 1e9 / static_cast<double>(solves);
    // companion timing (one pass; it is much slower)
    const auto t1 = Clock::now();
    for (const auto& p : polys) sink += CompanionRoots(p)[0].real();
    const double comp_ns =
        std::chrono::duration<double>(Clock::now() - t1).count() * 1e9 / 64.0;

    std::string heap = "n/a";
#ifdef RPOLY_BENCH_NO_MALLOC_GATE
    {
      // Pre-size the output vectors so a gated solve cannot allocate merely by
      // resizing its results, then forbid Eigen allocation for a single solve:
      // a heap fallback (VectorXd) trips the throwing gate, a stack-resident
      // solve (StackVec) does not.
      re = VectorXd::Zero(deg);
      im = VectorXd::Zero(deg);
      (void)FindPolynomialRootsJenkinsTraub(polys[0], &re, &im);  // warm
      bool allocated = false;
      Eigen::internal::set_is_malloc_allowed(false);
      try {
        (void)FindPolynomialRootsJenkinsTraub(polys[0], &re, &im);
      } catch (...) {
        allocated = true;
      }
      Eigen::internal::set_is_malloc_allowed(true);
      heap = allocated ? "no" : "yes";
      if (deg > 32) heap += " (heap fallback expected)";
      else if (allocated) heap += " (UNEXPECTED)";
    }
#endif
    std::printf("%8d %14.1f %16.1f %12s\n", deg, jt_ns, comp_ns, heap.c_str());
    (void)sink;
  }
}

// --------------------------- optional CSV corpus ---------------------------

void RunCorpus(const char* path) {
  std::ifstream in(path);
  if (!in) {
    std::printf("\n(corpus '%s' not readable; skipping)\n", path);
    return;
  }
  std::printf("\n== Field corpus: %s ==\n", path);
  std::string line;
  int n = 0, fails = 0;
  double res = 0;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<double> c;
    double x;
    while (ss >> x) {
      c.push_back(x);
      if (ss.peek() == ',') ss.ignore();
    }
    if (c.size() < 2) continue;
    VectorXd p(static_cast<int>(c.size()));
    for (size_t i = 0; i < c.size(); ++i) p(static_cast<int>(i)) = c[i];
    bool ok;
    const auto got = Solve(p, &ok);
    ++n;
    if (!ok) {
      ++fails;
      continue;
    }
    for (const auto& z : got) res = std::max(res, NormalizedResidual(p, z));
  }
  char nm[64];
  std::snprintf(nm, sizeof(nm), "corpus (%d rows): failures", n);
  Check(nm, fails, 0);
  std::snprintf(nm, sizeof(nm), "corpus (%d rows): residual", n);
  Check(nm, res, 1e-10);
}

// ============================================================================
// Cross-solver comparison: run every compiled-in solver over one shared corpus
// ============================================================================

struct CorpusCase {
  std::string family;
  VectorXd poly;
  std::vector<Complex> truth;  // analytic roots, or empty if unknown
};

std::vector<CorpusCase> BuildCorpus() {
  std::vector<CorpusCase> corpus;
  std::mt19937 g(20260715);

  {  // separated real deg-10
    std::uniform_real_distribution<double> u(-5.0, 5.0);
    std::vector<Complex> r;
    while (static_cast<int>(r.size()) < 10) {
      const double x = u(g);
      bool ok = true;
      for (const auto& y : r)
        if (std::abs(x - y.real()) < 0.25) ok = false;
      if (ok) r.emplace_back(x, 0.0);
    }
    corpus.push_back({"separated real d10", PolyFromRoots(r), r});
  }
  {  // complex conjugate pairs deg-10
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    std::vector<Complex> r;
    for (int i = 0; i < 5; ++i) {
      const double re = u(g), im = std::abs(u(g)) + 0.3;
      r.emplace_back(re, im);
      r.emplace_back(re, -im);
    }
    corpus.push_back({"complex pairs d10", PolyFromRoots(r), r});
  }
  {  // wide dynamic range 1e-4 .. 1e4
    std::vector<Complex> r;
    for (int k = -4; k <= 4; ++k) r.emplace_back(std::pow(10.0, k), 0.0);
    corpus.push_back({"wide range 1e-4..1e4", PolyFromRoots(r), r});
  }
  for (const int nw : {10, 15}) {  // Wilkinson
    std::vector<Complex> r;
    for (int i = 1; i <= nw; ++i) r.emplace_back(i, 0.0);
    char nm[32];
    std::snprintf(nm, sizeof(nm), "wilkinson-%d", nw);
    corpus.push_back({nm, PolyFromRoots(r), r});
  }
  {  // multiplicity ladder (x-1)^4 (x+2)
    std::vector<Complex> r(4, Complex(1, 0));
    r.emplace_back(-2, 0);
    corpus.push_back({"(x-1)^4 (x+2)", PolyFromRoots(r), r});
  }
  {  // cyclotomic x^20 - 1 (all roots on the unit circle)
    const int nc = 20;
    VectorXd p = VectorXd::Zero(nc + 1);
    p(0) = 1;
    p(nc) = -1;
    std::vector<Complex> r;
    for (int k = 0; k < nc; ++k)
      r.emplace_back(std::cos(2 * M_PI * k / nc), std::sin(2 * M_PI * k / nc));
    corpus.push_back({"cyclotomic x^20-1", p, r});
  }
  {  // Mignotte-9, a=100 (near-double cluster; no simple analytic truth)
    const int nm = 9;
    const double a = 100.0;
    VectorXd p = VectorXd::Zero(nm + 1);
    p(0) = 1;
    p(nm - 2) += -2 * a * a;
    p(nm - 1) += 4 * a;
    p(nm) += -2;
    corpus.push_back({"mignotte-9 a=100", p, {}});
  }
  {  // Kac deg-20 random coefficients (roots crowd the unit circle)
    std::normal_distribution<double> nd(0.0, 1.0);
    const int nk = 20;
    VectorXd p(nk + 1);
    for (int i = 0; i <= nk; ++i) p(i) = nd(g);
    if (std::abs(p(0)) < 1e-3) p(0) = 1.0;
    corpus.push_back({"kac deg-20", p, {}});
  }
  return corpus;
}

struct SolverEntry {
  std::string name;
  std::function<std::vector<Complex>(const VectorXd&, bool*)> fn;
  bool is_oracle;
};

struct Measure {
  double residual;
  double fwd_err;
  double ns;
  bool count_ok;  // returned the correct number of roots
  bool solved;    // count_ok AND backward residual under a uniform accept bar
  bool have_fwd;
};

void RunCrossSolverComparison(const std::vector<SolverEntry>& solvers) {
  std::printf("\n== Cross-solver comparison (shared corpus) ==\n");
  const auto corpus = BuildCorpus();

  // If an oracle is registered, precompute its roots per case to serve as the
  // forward-error reference where no analytic truth exists.
  int oracle_idx = -1;
  for (int s = 0; s < static_cast<int>(solvers.size()); ++s)
    if (solvers[s].is_oracle) oracle_idx = s;
  std::vector<std::vector<Complex>> oracle_roots(corpus.size());
  if (oracle_idx >= 0) {
    for (size_t c = 0; c < corpus.size(); ++c) {
      bool ok;
      oracle_roots[c] = solvers[oracle_idx].fn(corpus[c].poly, &ok);
    }
  }

  std::vector<std::vector<Measure>> R(
      solvers.size(), std::vector<Measure>(corpus.size()));

  for (size_t s = 0; s < solvers.size(); ++s) {
    for (size_t c = 0; c < corpus.size(); ++c) {
      const VectorXd& p = corpus[c].poly;
      bool ok = false;
      const auto roots = solvers[s].fn(p, &ok);
      (void)ok;  // each solver is judged by the same external yardstick below.
      Measure m;
      m.count_ok = static_cast<Eigen::Index>(roots.size()) == p.size() - 1;
      m.residual = 0.0;
      m.fwd_err = 0.0;
      m.have_fwd = false;
      for (const auto& z : roots)
        m.residual = std::max(m.residual, NormalizedResidual(p, z));
      // A "solve" = right number of roots, each a root to near rounding level.
      m.solved = m.count_ok && m.residual < 1e-6;
      const std::vector<Complex>* ref = nullptr;
      if (!corpus[c].truth.empty()) {
        ref = &corpus[c].truth;
      } else if (oracle_idx >= 0 && static_cast<int>(s) != oracle_idx &&
                 !oracle_roots[c].empty()) {
        ref = &oracle_roots[c];
      }
      if (ref != nullptr && roots.size() == ref->size()) {
        m.fwd_err = MaxRelForwardError(roots, *ref);
        m.have_fwd = true;
      }
      // Timing: repeat the solve until at least 5 ms have elapsed.
      const auto t0 = Clock::now();
      long reps = 0;
      double el = 0.0;
      volatile double sink = 0.0;
      do {
        bool tok;
        const auto rr = solvers[s].fn(p, &tok);
        if (!rr.empty()) sink += rr[0].real();
        ++reps;
        el = std::chrono::duration<double>(Clock::now() - t0).count();
      } while (el < 0.005 && reps < 100000);
      (void)sink;
      m.ns = el * 1e9 / static_cast<double>(reps);
      R[s][c] = m;
    }
  }

  std::printf("\n  %-18s %6s %6s %13s %13s %14s\n", "solver", "cases", "fails",
              "max resid", "max fwd err", "ns/solve(mean)");
  for (size_t s = 0; s < solvers.size(); ++s) {
    int fails = 0;
    double mr = 0, mf = 0, tot = 0;
    for (size_t c = 0; c < corpus.size(); ++c) {
      if (!R[s][c].solved) ++fails;
      if (R[s][c].count_ok) mr = std::max(mr, R[s][c].residual);
      if (R[s][c].have_fwd && R[s][c].count_ok)
        mf = std::max(mf, R[s][c].fwd_err);
      tot += R[s][c].ns;
    }
    std::printf("  %-18s %6zu %6d %13.3e %13.3e %14.1f%s\n",
                solvers[s].name.c_str(), corpus.size(), fails, mr, mf,
                tot / static_cast<double>(corpus.size()),
                solvers[s].is_oracle ? "  (oracle)" : "");
  }

  std::printf("\n  per-family max backward residual (FAIL = wrong root count):\n");
  std::printf("  %-22s", "family");
  for (const auto& sv : solvers) std::printf(" %12.12s", sv.name.c_str());
  std::printf("\n");
  for (size_t c = 0; c < corpus.size(); ++c) {
    std::printf("  %-22.22s", corpus[c].family.c_str());
    for (size_t s = 0; s < solvers.size(); ++s) {
      if (R[s][c].count_ok)
        std::printf(" %12.2e", R[s][c].residual);
      else
        std::printf(" %12s", "FAIL");
    }
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("RpolyPlusPlus comprehensive evaluation\n");
  std::printf("======================================\n");

  RunClassicFamilies();
  RunP3P();
  RunSevenPoint();
  RunTorus();
  RunEllipse();
  RunDifferentialAndSpeed();

  std::vector<SolverEntry> solvers;
  solvers.push_back({"Jenkins-Traub",
                     [](const VectorXd& p, bool* ok) {
                       VectorXd re, im;
                       *ok = FindPolynomialRootsJenkinsTraub(p, &re, &im);
                       std::vector<Complex> r;
                       if (*ok)
                         for (int i = 0; i < re.size(); ++i)
                           r.emplace_back(re(i), im(i));
                       return r;
                     },
                     false});
  solvers.push_back({"Companion QR",
                     [](const VectorXd& p, bool* ok) {
                       *ok = true;
                       return CompanionRoots(p);
                     },
                     false});
  solvers.push_back({"Aberth-Ehrlich", rpoly_bench::AberthSolve, false});
#ifdef WITH_RPOLY
  solvers.push_back({"RPOLY (AK1)", rpoly_bench::RpolyAk1Solve, false});
#endif
#ifdef WITH_MPSOLVE
  solvers.push_back({"MPSolve", rpoly_bench::MpsolveSolve, true});
#endif
  RunCrossSolverComparison(solvers);

  if (argc > 1) RunCorpus(argv[1]);

  int failed = 0;
  for (const auto& gt : Gates())
    if (!gt.pass) ++failed;
  std::printf("\n================ SUMMARY ================\n");
  std::printf("gates: %zu   passed: %zu   FAILED: %d\n", Gates().size(),
              Gates().size() - static_cast<size_t>(failed), failed);
  if (failed) {
    for (const auto& gt : Gates())
      if (!gt.pass)
        std::printf("  FAIL: %s  (%.3e > %.1e)\n", gt.name.c_str(),
                    gt.measured, gt.threshold);
  }
  return failed;
}
