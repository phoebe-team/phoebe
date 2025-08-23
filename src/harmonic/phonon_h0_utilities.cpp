#include <cmath>
#include <iostream>

#include "crystal.h"
#include "eigen.h"
#include "exceptions.h"
#include "utilities.h"
#include "common_kokkos.h"
#include "mpiHelper.h"
#include "phonon_h0.h"

/** Auxiliary methods for sum rule on Born charges
 */
void sp_zeu(Eigen::Tensor<double, 3> &zeu_u,
                      Eigen::Tensor<double, 3> &zeu_v, double &scalar, int& numAtoms) {
  // get the scalar product of two effective charges matrices zeu_u and zeu_v
  // (considered as vectors in the R^(3*3*nat) space)

  scalar = 0.;
#pragma omp parallel for collapse(3) reduction(+ : scalar)
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int na = 0; na < numAtoms; na++) {
        scalar += zeu_u(i, j, na) * zeu_v(i, j, na);
      }
    }
  }
}

void setAcousticSumRule(const std::string &sumRule, Crystal& crystal, 
                        const Eigen::Vector3i& qCoarseGrid,
                        Eigen::Tensor<double, 7>& forceConstants) {

  //  VectorXi u_less(6*3*numAtoms)
  //  indices of the vectors u that are not independent to the preceding ones
  //  n_less = number of such vectors
  //
  //  Tensor<int> ind_v(:,:,:)
  //  Tensor<double> v(:,:)
  //  These are the "vectors" associated with symmetry conditions, coded by
  //  indicating the positions (i.e. the seven indices) of the non-zero
  //  elements (there should be only 2 of them) and the value of that element
  //  We do so in order to limit the amount of memory used.
  //
  //  Tensor<double> zeu_u(6*3,3,3,numAtoms)
  //  These are vectors associated with the sum rules on effective charges
  //
  //  Tensor<int> zeu_less(6*3)
  //  indices of zeu_u vectors that are not independent to the preceding ones
  //  ! nzeu_less = number of such vectors

  int numAtoms = crystal.getNumAtoms();
  auto bornCharges = crystal.getBornEffectiveCharges();

  std::string sr = sumRule;
  std::transform(sr.begin(), sr.end(), sr.begin(), ::tolower);

  if (sr.empty()) {
    return;
  }

  if ((sr != "simple") && (sr != "crystal")) {
    Error("invalid Acoustic Sum Rule");
  }

  if (mpi->mpiHead()) {
    std::cout << "Start imposing " << sumRule << " acoustic sum rule."
              << std::endl;
  }

  if (sr == "simple") {
    // Simple Acoustic Sum Rule on effective charges

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        double sum = 0.;
#pragma omp parallel for reduction(+ : sum)
        for (int na = 0; na < numAtoms; na++) {
          sum += bornCharges(na, i, j);
        }
        for (int na = 0; na < numAtoms; na++) {
          bornCharges(na, i, j) -= sum / numAtoms;
        }
      }
    }

    // Simple Acoustic Sum Rule on force constants in real space

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int na = 0; na < numAtoms; na++) {
          double sum = 0.;
          for (int nb = 0; nb < numAtoms; nb++) {
            for (int n1 = 0; n1 < qCoarseGrid(0); n1++) {
              for (int n2 = 0; n2 < qCoarseGrid(1); n2++) {
                for (int n3 = 0; n3 < qCoarseGrid(2); n3++) {
                  sum += forceConstants(i, j, n1, n2, n3, na, nb);
                }
              }
            }
          }
          forceConstants(i, j, 0, 0, 0, na, na) -= sum;
        }
      }
    }
  } else {
    // Acoustic Sum Rule on effective charges

    // generating the vectors of the orthogonal of the subspace to project
    // the effective charges matrix on

    Eigen::Tensor<double, 4> zeu_u(6 * 3, 3, 3, numAtoms);
    zeu_u.setZero();
    Eigen::Tensor<double, 3> zeu_new(3, 3, numAtoms);
    zeu_new.setZero();

#pragma omp parallel for collapse(3)
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int iat = 0; iat < numAtoms; iat++) {
          zeu_new(i, j, iat) = bornCharges(iat, i, j);
        }
      }
    }

    int p = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
#pragma omp parallel for
        for (int iat = 0; iat < numAtoms; iat++) {
          // These are the 3*3 vectors associated with the
          // translational acoustic sum rules
          zeu_u(p, i, j, iat) = 1.;
        }
        p += 1;
      }
    }

    // Gram-Schmidt orto-normalization of the set of vectors created.

    // temporary vectors
    Eigen::Tensor<double, 3> zeu_w(3, 3, numAtoms), zeu_x(3, 3, numAtoms);
    Eigen::Tensor<double, 3> tempZeu(3, 3, numAtoms);
    // note: it's important to initialize these tensors
    zeu_w.setZero();
    zeu_x.setZero();
    tempZeu.setZero();
    Eigen::VectorXi zeu_less(6 * 3);
    zeu_less.setZero();
    double scalar;
    int nzeu_less = 0;
    int r;

    for (int k = 0; k < p; k++) {
#pragma omp parallel for collapse(3)
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int iat = 0; iat < numAtoms; iat++) {
            zeu_w(i, j, iat) = zeu_u(k, i, j, iat);
            zeu_x(i, j, iat) = zeu_u(k, i, j, iat);
          }
        }
      }

      for (int q = 0; q < k - 1; q++) {
        r = 1;
        for (int iZeu_less = 0; iZeu_less < nzeu_less; iZeu_less++) {
          if (zeu_less(iZeu_less) == q) {
            r = 0;
          }
        }
        if (r != 0) {
#pragma omp parallel for collapse(3)
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              for (int iat = 0; iat < numAtoms; iat++) {
                tempZeu(i, j, iat) = zeu_u(q, i, j, iat);
              }
            }
          }
          // i.e. zeu_u(q,:,:,:)
          sp_zeu(zeu_x, tempZeu, scalar, numAtoms);
          zeu_w -= scalar * tempZeu;
        }
      }
      double norm2;
      sp_zeu(zeu_w, zeu_w, norm2, numAtoms);

      if (norm2 > 1.0e-16) {
#pragma omp parallel for collapse(3)
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            for (int iat = 0; iat < numAtoms; iat++) {
              zeu_u(k, i, j, iat) = zeu_w(i, j, iat) / sqrt(norm2);
            }
          }
        }
      } else {
        zeu_less(nzeu_less) = k;
        nzeu_less += 1;
      }
    }

    // Projection of the effective charge "vector" on the orthogonal of the
    // subspace of the vectors verifying the sum rules

    zeu_w.setZero();
    for (int k = 0; k < p; k++) {
      r = 1;
      for (int izeu_less = 0; izeu_less < nzeu_less; izeu_less++) {
        if (zeu_less(izeu_less) == k) {
          r = 0;
        }
      }
      if (r != 0) {
        // copy vector
#pragma omp parallel for collapse(3)
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            for (int iat = 0; iat < numAtoms; iat++) {
              zeu_x(i, j, iat) = zeu_u(k, i, j, iat);
            }
          }
        }
        // get rescaling factor
        sp_zeu(zeu_x, zeu_new, scalar, numAtoms);
        // rescale vector

        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            for (int iat = 0; iat < numAtoms; iat++) {
              zeu_w(i, j, iat) += scalar * zeu_u(k, i, j, iat);
            }
          }
        }
      }
    }

    // Final subtraction of the former projection to the initial zeu, to
    // get the new "projected" zeu

    zeu_new -= zeu_w;
    double norm2;
    sp_zeu(zeu_w, zeu_w, norm2, numAtoms);
    if (mpi->mpiHead()) {
      std::cout << "Norm of the difference between old and new effective "
                   "charges: "
                << sqrt(norm2) << std::endl;
    }

#pragma omp parallel for collapse(3)
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int iat = 0; iat < numAtoms; iat++) {
          bornCharges(iat, i, j) = zeu_new(i, j, iat);
        }
      }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Acoustic Sum Rule on force constants

    // generating the vectors of the orthogonal of the subspace to project
    // the force-constants matrix on

    int nr1 = qCoarseGrid(0);
    int nr2 = qCoarseGrid(1);
    int nr3 = qCoarseGrid(2);

    Eigen::Tensor<double, 8> uvec(18 * numAtoms, nr1, nr2, nr3, 3, 3, numAtoms,
                                  numAtoms);
    uvec.setZero();

    Eigen::Tensor<double, 7> frcNew(nr1, nr2, nr3, 3, 3, numAtoms, numAtoms);
#pragma omp parallel for collapse(7)
    for (int nb = 0; nb < numAtoms; nb++) {
      for (int na = 0; na < numAtoms; na++) {
        for (int j = 0; j < 3; j++) {
          for (int i = 0; i < 3; i++) {
            for (int n3 = 0; n3 < nr3; n3++) {
              for (int n2 = 0; n2 < nr2; n2++) {
                for (int n1 = 0; n1 < nr1; n1++) {
                  frcNew(n1, n2, n3, i, j, na, nb) =
                      forceConstants(i, j, n1, n2, n3, na, nb);
                }
              }
            }
          }
        }
      }
    }

    p = 0;
    for (int na = 0; na < numAtoms; na++) {
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
          // These are the 3*3*nat vectors associated with the
          // translational acoustic sum rules
#pragma omp parallel for collapse(4)
          for (int nb = 0; nb < numAtoms; nb++) {
            for (int n3 = 0; n3 < nr3; n3++) {
              for (int n2 = 0; n2 < nr2; n2++) {
                for (int n1 = 0; n1 < nr1; n1++) {
                  uvec(p, n1, n2, n3, i, j, na, nb) = 1.;
                }
              }
            }
          }
          p += 1;
        }
      }
    }

    Eigen::Tensor<int, 3> ind_v(9 * numAtoms * numAtoms * nr1 * nr2 * nr3, 2,
                                7);
    ind_v.setZero();
    Eigen::Tensor<double, 2> v(9 * numAtoms * numAtoms * nr1 * nr2 * nr3, 2);
    v.setZero();

    int m = 0;
    int q, l;

    for (int i : {1, 2, 3}) {
      for (int j : {1, 2, 3}) {
        for (int na = 1; na <= numAtoms; na++) {
          for (int nb = 1; nb <= numAtoms; nb++) {
            for (int n1 = 1; n1 <= nr1; n1++) {
              for (int n2 = 1; n2 <= nr2; n2++) {
                for (int n3 = 1; n3 <= nr3; n3++) {
                  // These are the vectors associated with the symmetry
                  // constraints
                  q = 1;
                  l = 1;
                  while ((l <= m) && (q != 0)) {
                    if ((ind_v(l - 1, 0, 0) == n1) &&
                        (ind_v(l - 1, 0, 1) == n2) &&
                        (ind_v(l - 1, 0, 2) == n3) &&
                        (ind_v(l - 1, 0, 3) == i) &&
                        (ind_v(l - 1, 0, 4) == j) &&
                        (ind_v(l - 1, 0, 5) == na) &&
                        (ind_v(l - 1, 0, 6) == nb)) {
                      q = 0;
                    }
                    if ((ind_v(l - 1, 1, 0) == n1) &&
                        (ind_v(l - 1, 1, 1) == n2) &&
                        (ind_v(l - 1, 1, 2) == n3) &&
                        (ind_v(l - 1, 1, 3) == i) &&
                        (ind_v(l - 1, 1, 4) == j) &&
                        (ind_v(l - 1, 1, 5) == na) &&
                        (ind_v(l - 1, 1, 6) == nb)) {
                      q = 0;
                    }
                    l += 1;
                  }
                  if ((n1 == mod((nr1 + 1 - n1) , nr1) + 1) &&
                      (n2 == mod((nr2 + 1 - n2) , nr2) + 1) &&
                      (n3 == mod((nr3 + 1 - n3) , nr3) + 1) && (i == j) &&
                      (na == nb)) {
                    q = 0;
                  }
                  if (q != 0) {
                    m += 1;
                    ind_v(m - 1, 0, 0) = n1;
                    ind_v(m - 1, 0, 1) = n2;
                    ind_v(m - 1, 0, 2) = n3;
                    ind_v(m - 1, 0, 3) = i;
                    ind_v(m - 1, 0, 4) = j;
                    ind_v(m - 1, 0, 5) = na;
                    ind_v(m - 1, 0, 6) = nb;
                    v(m - 1, 0) = 1. / sqrt(2.);
                    ind_v(m - 1, 1, 0) = mod((nr1 + 1 - n1) , nr1) + 1;
                    ind_v(m - 1, 1, 1) = mod((nr2 + 1 - n2) , nr2) + 1;
                    ind_v(m - 1, 1, 2) = mod((nr3 + 1 - n3) , nr3) + 1;
                    ind_v(m - 1, 1, 3) = j;
                    ind_v(m - 1, 1, 4) = i;
                    ind_v(m - 1, 1, 5) = nb;
                    ind_v(m - 1, 1, 6) = na;
                    v(m - 1, 1) = -1. / sqrt(2.);
                  }
                }
              }
            }
          }
        }
      }
    }

    // Gram-Schmidt orto-normalization of the set of vectors created.
    // Note that the vectors corresponding to symmetry constraints are already
    // orthonormalized by construction.

    int n_less = 0;
    Eigen::Tensor<double, 7> w(nr1, nr2, nr3, 3, 3, numAtoms, numAtoms);
    Eigen::Tensor<double, 7> x(nr1, nr2, nr3, 3, 3, numAtoms, numAtoms);
    w.setZero();
    x.setZero();

    Eigen::VectorXi u_less(6 * 3 * numAtoms);
    u_less.setZero();

    for (int k = 1; k <= p; k++) {
#pragma omp parallel for collapse(7)
      for (int nb = 0; nb < numAtoms; nb++) {
        for (int na = 0; na < numAtoms; na++) {
          for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
              for (int n3 = 0; n3 < nr3; n3++) {
                for (int n2 = 0; n2 < nr2; n2++) {
                  for (int n1 = 0; n1 < nr1; n1++) {
                    w(n1, n2, n3, i, j, na, nb) =
                        uvec(k - 1, n1, n2, n3, i, j, na, nb);
                    x(n1, n2, n3, i, j, na, nb) =
                        uvec(k - 1, n1, n2, n3, i, j, na, nb);
                  }
                }
              }
            }
          }
        }
      }

      for (l = 0; l < m; l++) {
        scalar = 0.;
        for (int rr : {0, 1}) {
          int n1 = ind_v(l, rr, 0) - 1;
          int n2 = ind_v(l, rr, 1) - 1;
          int n3 = ind_v(l, rr, 2) - 1;
          int i = ind_v(l, rr, 3) - 1;
          int j = ind_v(l, rr, 4) - 1;
          int na = ind_v(l, rr, 5) - 1;
          int nb = ind_v(l, rr, 6) - 1;
          scalar += w(n1, n2, n3, i, j, na, nb) * v(l, rr);
        }

        for (int rr : {0, 1}) {
          int n1 = ind_v(l, rr, 0) - 1;
          int n2 = ind_v(l, rr, 1) - 1;
          int n3 = ind_v(l, rr, 2) - 1;
          int i = ind_v(l, rr, 3) - 1;
          int j = ind_v(l, rr, 4) - 1;
          int na = ind_v(l, rr, 5) - 1;
          int nb = ind_v(l, rr, 6) - 1;
          w(n1, n2, n3, i, j, na, nb) -= scalar * v(l, rr);
        }
      }

      int i1, j1, na1;
      if (k <= (9 * numAtoms)) {
        na1 = mod(k , numAtoms);
        if (na1 == 0) {
          na1 = numAtoms;
        }
        j1 = mod(((k - na1) / numAtoms) , 3) + 1;
        i1 = mod(((((k - na1) / numAtoms) - j1 + 1) / 3) , 3) + 1;
      } else {
        q = k - 9 * numAtoms;
        na1 = mod(q , numAtoms);
        if (na1 == 0)
          na1 = numAtoms;
        j1 = mod(((q - na1) / numAtoms) , 3) + 1;
        i1 = mod(((((q - na1) / numAtoms) - j1 + 1) / 3) , 3) + 1;
      }
      for (q = 1; q <= k - 1; q++) {
        r = 1;
        for (int i_less = 1; i_less <= n_less; i_less++) {
          if (u_less(i_less - 1) == q) {
            r = 0;
          }
        }
        if (r != 0) {
          scalar = 0.;
#pragma omp parallel for collapse(5) reduction(+ : scalar)
          for (int nb = 0; nb < numAtoms; nb++) {
            for (int j = 0; j < 3; j++) {
              for (int n3 = 0; n3 < nr3; n3++) {
                for (int n2 = 0; n2 < nr2; n2++) {
                  for (int n1 = 0; n1 < nr1; n1++) {
                    scalar += x(n1, n2, n3, i1 - 1, j, na1 - 1, nb) *
                              uvec(q - 1, n1, n2, n3, i1 - 1, j, na1 - 1, nb);
                  }
                }
              }
            }
          }

          for (int nb = 0; nb < numAtoms; nb++) {
            for (int na = 0; na < numAtoms; na++) {
              for (int j = 0; j < 3; j++) {
                for (int i = 0; i < 3; i++) {
                  for (int n3 = 0; n3 < nr3; n3++) {
                    for (int n2 = 0; n2 < nr2; n2++) {
                      for (int n1 = 0; n1 < nr1; n1++) {
                        w(n1, n2, n3, i, j, na, nb) -=
                            scalar * uvec(q - 1, n1, n2, n3, i, j, na, nb);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      norm2 = 0.;
#pragma omp parallel for collapse(7) reduction(+ : norm2)
      for (int nb = 0; nb < numAtoms; nb++) {
        for (int na = 0; na < numAtoms; na++) {
          for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
              for (int n3 = 0; n3 < nr3; n3++) {
                for (int n2 = 0; n2 < nr2; n2++) {
                  for (int n1 = 0; n1 < nr1; n1++) {
                    norm2 += w(n1, n2, n3, i, j, na, nb) *
                             w(n1, n2, n3, i, j, na, nb);
                  }
                }
              }
            }
          }
        }
      }

      if (norm2 > 1.0e-16) {
#pragma omp parallel for collapse(7)
        for (int nb = 0; nb < numAtoms; nb++) {
          for (int na = 0; na < numAtoms; na++) {
            for (int j = 0; j < 3; j++) {
              for (int i = 0; i < 3; i++) {
                for (int n3 = 0; n3 < nr3; n3++) {
                  for (int n2 = 0; n2 < nr2; n2++) {
                    for (int n1 = 0; n1 < nr1; n1++) {
                      uvec(k - 1, n1, n2, n3, i, j, na, nb) =
                          w(n1, n2, n3, i, j, na, nb) / sqrt(norm2);
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        n_less += 1;
        u_less(n_less - 1) = k;
      }
    }

    // Projection of the force-constants "vector" on the orthogonal of the
    // subspace of the vectors verifying the sum rules and symmetry constraints

    w.setZero();
    for (l = 1; l <= m; l++) {

      //      call sp2(frcNew,v(l,:),ind_v(l,:,:),nr1,nr2,nr3,nat,scalar)
      scalar = 0.;
      for (int ii : {0, 1}) {
        int n1 = ind_v(l - 1, ii, 0) - 1;
        int n2 = ind_v(l - 1, ii, 1) - 1;
        int n3 = ind_v(l - 1, ii, 2) - 1;
        int i = ind_v(l - 1, ii, 3) - 1;
        int j = ind_v(l - 1, ii, 4) - 1;
        int na = ind_v(l - 1, ii, 5) - 1;
        int nb = ind_v(l - 1, ii, 6) - 1;
        scalar += frcNew(n1, n2, n3, i, j, na, nb) * v(l - 1, ii);
      }

      for (int rr : {0, 1}) {
        int n1 = ind_v(l - 1, rr, 0) - 1;
        int n2 = ind_v(l - 1, rr, 1) - 1;
        int n3 = ind_v(l - 1, rr, 2) - 1;
        int i = ind_v(l - 1, rr, 3) - 1;
        int j = ind_v(l - 1, rr, 4) - 1;
        int na = ind_v(l - 1, rr, 5) - 1;
        int nb = ind_v(l - 1, rr, 6) - 1;
        w(n1, n2, n3, i, j, na, nb) += scalar * v(l - 1, rr);
      }
    }
    for (int k = 1; k <= p; k++) {
      r = 1;
      for (int i_less = 1; i_less <= n_less; i_less++) {
        if (u_less(i_less - 1) == k) {
          r = 0;
        }
      }
      if (r != 0) {

        scalar = 0.;
#pragma omp parallel for collapse(7) reduction(+ : scalar)
        for (int nb = 0; nb < numAtoms; nb++) {
          for (int na = 0; na < numAtoms; na++) {
            for (int j = 0; j < 3; j++) {
              for (int i = 0; i < 3; i++) {
                for (int n3 = 0; n3 < nr3; n3++) {
                  for (int n2 = 0; n2 < nr2; n2++) {
                    for (int n1 = 0; n1 < nr1; n1++) {
                      scalar += uvec(k - 1, n1, n2, n3, i, j, na, nb) * frcNew(n1, n2, n3, i, j, na, nb);
                    }
                  }
                }
              }
            }
          }
        }

        for (int nb = 0; nb < numAtoms; nb++) {
          for (int na = 0; na < numAtoms; na++) {
            for (int j = 0; j < 3; j++) {
              for (int i = 0; i < 3; i++) {
                for (int n3 = 0; n3 < nr3; n3++) {
                  for (int n2 = 0; n2 < nr2; n2++) {
                    for (int n1 = 0; n1 < nr1; n1++) {
                      w(n1, n2, n3, i, j, na, nb) +=
                          scalar * uvec(k - 1, n1, n2, n3, i, j, na, nb);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // Final subtraction of the former projection to the initial frc,
    // to get the new "projected" frc

    frcNew -= w;
    scalar = 0.;
#pragma omp parallel for reduction(+ : scalar) collapse(7)
    for (int nb = 0; nb < numAtoms; nb++) {
      for (int na = 0; na < numAtoms; na++) {
        for (int j = 0; j < 3; j++) {
          for (int i = 0; i < 3; i++) {
            for (int n3 = 0; n3 < nr3; n3++) {
              for (int n2 = 0; n2 < nr2; n2++) {
                for (int n1 = 0; n1 < nr1; n1++) {
                  scalar +=
                      w(n1, n2, n3, i, j, na, nb) * w(n1, n2, n3, i, j, na, nb);
                }
              }
            }
          }
        }
      }
    }

    if (mpi->mpiHead()) {
      std::cout << "Norm of the difference between old and new "
                   "force-constants: "
                << sqrt(scalar) << std::endl;
    }

#pragma omp parallel for collapse(7)
    for (int nb = 0; nb < numAtoms; nb++) {
      for (int na = 0; na < numAtoms; na++) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            for (int n3 = 0; n3 < nr3; n3++) {
              for (int n2 = 0; n2 < nr2; n2++) {
                for (int n1 = 0; n1 < nr1; n1++) {
                  forceConstants(i, j, n1, n2, n3, na, nb) =
                      frcNew(n1, n2, n3, i, j, na, nb);
                }
              }
            }
          }
        }
      }
    }
  }
  if (mpi->mpiHead()) {
    std::cout << "Finished imposing " << sumRule << " acoustic sum rule."
              << std::endl;
  }
}

  /** wsWeight computes the `weights`, i.e. the number of symmetry-equivalent
   * Bravais lattice vectors, that are used in the phonon Fourier transform.
   */
double wsWeight(const Eigen::VectorXd &r, const Eigen::MatrixXd &rws) {
  // wsWeight() assigns this weight:
  // - if a point is inside the Wigner-Seitz cell:    weight=1
  // - if a point is outside the WS cell:             weight=0
  // - if a point q is on the border of the WS cell, it finds the number N
  //   of translationally equivalent point q+G  (where G is a lattice vector)
  //   that are also on the border of the cell. Then: weight = 1/N
  //
  // I.e. if a point is on the surface of the WS cell of a cubic lattice
  // it will have weight 1/2; on the vertex of the WS it would be 1/8;
  // the K point of an hexagonal lattice has weight 1/3 and so on.

  // rws: contains the list of nearest neighbor atoms
  // r: the position of the reference point
  // rws.cols(): number of nearest neighbors

  int numREq = 1;

  for (int ir = 0; ir < rws.cols(); ir++) {
    double rrt = r.dot(rws.col(ir));
    double ck = rrt - rws.col(ir).squaredNorm() / 2.;
    if (ck > 1.0e-6) {
      return 0.;
    }
    if (abs(ck) <= 1.0e-6) {
      numREq += 1;
    }
  }
  double x = 1. / (double)numREq;
  return x;
}

std::tuple<Eigen::Tensor<double, 5>,Eigen::MatrixXd,Eigen::VectorXd> 
          reorderHarmonicForceConstants(Crystal& crystal,
                                const Eigen::Tensor<double, 7>& forceConstants,
                                Eigen::Vector3i& qCoarseGrid) {

  Kokkos::Profiling::pushRegion("reorderForceConstants");

  // this part can actually be expensive to execute, so we compute it once
  // at the beginning
  auto directUnitCell = crystal.getDirectUnitCell();
  auto atomicPositions = crystal.getAtomicPositions();
  int numAtoms = crystal.getNumAtoms();

  Eigen::MatrixXd directUnitCellSup(3, 3);
  directUnitCellSup.col(0) = directUnitCell.col(0) * qCoarseGrid(0);
  directUnitCellSup.col(1) = directUnitCell.col(1) * qCoarseGrid(1);
  directUnitCellSup.col(2) = directUnitCell.col(2) * qCoarseGrid(2);

  int nr1Big = 2 * qCoarseGrid(0);
  int nr2Big = 2 * qCoarseGrid(1);
  int nr3Big = 2 * qCoarseGrid(2);

  // start by generating the weights for the Fourier transform
  //auto wsCache = wsInit(directUnitCellSup, directUnitCell, nr1Big, nr2Big, nr3Big);

  // TODO the whole function inside these blocks should be replaced by buildWSVectorsWithShift
  // from crystal class, to be consistent. 
  // We would have to look up the index of each R vector in the list of iR provided by the
  // crystal class function for each vector inside the loops below. 
  // -----------------------------------------------
  const int numMax = 2;
  int index = 0;
  const int numMaxRWS = 200;

  Eigen::MatrixXd tmpResult(3, numMaxRWS);

  for (int ir = -numMax; ir <= numMax; ir++) {
    for (int jr = -numMax; jr <= numMax; jr++) {
      for (int kr = -numMax; kr <= numMax; kr++) {
        for (int i : {0, 1, 2}) {
          tmpResult(i, index) =
              directUnitCellSup(i, 0) * ir + directUnitCellSup(i, 1) * jr + directUnitCellSup(i, 2) * kr;
        }

        if (tmpResult.col(index).squaredNorm() > 1.0e-6) {
          index += 1;
        }
        if (index > numMaxRWS) {
          Error("WSInit > numMaxRWS");
        }
      }
    }
  }
  int numRWS = index;
  Eigen::MatrixXd rws(3, numRWS);
  for (int i = 0; i < numRWS; i++) {
    rws.col(i) = tmpResult.col(i);
  }

  // now, I also prepare the wsCache, which is used to accelerate
  // the shortRange() calculation

  Eigen::Tensor<double, 5> wsCache(2 * nr3Big + 1, 2 * nr2Big + 1,
                                   2 * nr1Big + 1, numAtoms, numAtoms);
  wsCache.setZero();

  for (int na = 0; na < numAtoms; na++) {
    for (int nb = 0; nb < numAtoms; nb++) {
      double total_weight = 0.;

      // sum over r vectors in the super cell - very safe range!

      for (int n1 = -nr1Big; n1 <= nr1Big; n1++) {
        int n1ForCache = n1 + nr1Big;
        for (int n2 = -nr2Big; n2 <= nr2Big; n2++) {
          int n2ForCache = n2 + nr2Big;
          for (int n3 = -nr3Big; n3 <= nr3Big; n3++) {
            int n3ForCache = n3 + nr3Big;

            Eigen::Vector3d r_ws;
            for (int i : {0, 1, 2}) {
              // note that this cell is different from above
              r_ws(i) = double(n1) * directUnitCell(i, 0) +
                        double(n2) * directUnitCell(i, 1) +
                        double(n3) * directUnitCell(i, 2);
              r_ws(i) += atomicPositions(na, i) - atomicPositions(nb, i);
            }

            double x = wsWeight(r_ws, rws);
            wsCache(n3ForCache, n2ForCache, n1ForCache, nb, na) = x;
            total_weight += x;
          }
        }
      }
      if (abs(total_weight - qCoarseGrid(0) * qCoarseGrid(1) * qCoarseGrid(2)) > 1.0e-8) {
        Error("DeveloperError: wrong total_weight, weight: "
                + std::to_string(total_weight) + " qMeshProd: " + std::to_string(qCoarseGrid.prod()) );
      }
    }
  }
  // ------------------------------------------


  // we compute the total number of bravais lattice vectors
  int numBravaisVectors = 0;
  for (int n3 = -nr3Big; n3 <= nr3Big; n3++) {
    int n3ForCache = n3 + nr3Big;
    for (int n2 = -nr2Big; n2 <= nr2Big; n2++) {
      int n2ForCache = n2 + nr2Big;
      for (int n1 = -nr1Big; n1 <= nr1Big; n1++) {
        int n1ForCache = n1 + nr1Big;
        for (int nb = 0; nb < numAtoms; nb++) {
          for (int na = 0; na < numAtoms; na++) {
            if (wsCache(n3ForCache, n2ForCache, n1ForCache, nb, na) > 0.) {
              numBravaisVectors += 1;
            }
          }
        }
      }
    }
  }

  // next, we reorder the dynamical matrix along the bravais lattice vectors 
  // and save the weights and vectors for later use 
  Eigen::MatrixXd bravaisVectors = Eigen::MatrixXd::Zero(3, numBravaisVectors);
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(numBravaisVectors);
  Eigen::Tensor<double,5> mat2R(3, 3, numAtoms, numAtoms, numBravaisVectors);
  mat2R.setZero();

  int iR = 0;
  for (int n3 = -nr3Big; n3 <= nr3Big; n3++) {
    int n3ForCache = n3 + nr3Big;
    for (int n2 = -nr2Big; n2 <= nr2Big; n2++) {
      int n2ForCache = n2 + nr2Big;
      for (int n1 = -nr1Big; n1 <= nr1Big; n1++) {
        int n1ForCache = n1 + nr1Big;
        // loop over the shifted vectors (R+T, in which we shift the origin to the atomic positions)
        for (int nb = 0; nb < numAtoms; nb++) {
          for (int na = 0; na < numAtoms; na++) {
            double weight = wsCache(n3ForCache, n2ForCache, n1ForCache, nb, na);
            if (weight > 0.) {
              weights(iR) = weight;

              Eigen::Vector3d r;
              for (int i : {0, 1, 2}) {
                r(i) = n1 * directUnitCell(i, 0) + n2 * directUnitCell(i, 1) +
                       n3 * directUnitCell(i, 2);
              }
              bravaisVectors.col(iR) = r;

              int m1 = mod((n1 + 1), qCoarseGrid(0));
              if (m1 <= 0) {
                m1 += qCoarseGrid(0);
              }
              int m2 = mod((n2 + 1), qCoarseGrid(1));
              if (m2 <= 0) {
                m2 += qCoarseGrid(1);
              }
              int m3 = mod((n3 + 1), qCoarseGrid(2));
              if (m3 <= 0) {
                m3 += qCoarseGrid(2);
              }
              m1 += -1;
              m2 += -1;
              m3 += -1;

              for (int j : {0, 1, 2}) {
                for (int i : {0, 1, 2}) {
                  // convert to a single iR vector index 
                  mat2R(i, j, na, nb, iR) += forceConstants(i, j, m1, m2, m3, na, nb);
                }
              }
              iR += 1;
            }
          }
        }
      }
    }
  }
  Kokkos::Profiling::popRegion();

  return std::make_tuple(mat2R,bravaisVectors,weights);

}