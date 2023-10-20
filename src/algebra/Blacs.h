#ifndef BLACS_H
#define BLACS_H
/** \file   blacs.h
 *  \brief  Header file for forward declaration of BLACS and SCALAPACK
 *  functions.
 */

#include <complex>

extern "C" {

void blacs_get_(int *, int *, int *);
void blacs_pinfo_(int *, int *);
void blacs_gridinit_(int *, char *, int *, int *);
void blacs_gridinfo_(int *, int *, int *, int *, int *);
void descinit_(int *, int *, int *, int *, int *, int *, int *, int *, int *,
               int *);
void blacs_gridexit_(const int *);
int numroc_(int *, int *, int *, int *, int *);

//void pdelset_(double *, int *, int *, int *, double *);
//void pdelget_(char *, char *, double *, double *, const int *, const int *,
//              const int *);
void infog2l_(const int *, const int *, const int *, const int *, const int *,
              const int *, const int *, int *, int *, int *, int *);

int indxg2p_(int *, int *, int *, int *, int *);

int indxg2l_(const int *, const int *, const int *, const int *, const int *);

int indxl2g_(const int *, const int *, const int *, const int *, const int *);

void pdgemm_(const char *, const char *, int *, int *, const int *, double *,
             double *, int *, int *, const int *, double *, int *, int *,
             const int *, double *, double *, int *, int *, int *);
void pdsyev_(char *, char *, int *, double *, int *, int *, int *, double *,
             double *, int *, int *, int *, double *, int *, int *);
//void pzelset_(std::complex<double> *, int *, int *, int *,
//              std::complex<double> *);
//void pzelget_(char *, char *, std::complex<double> *, std::complex<double> *,
//              int *, int *, int *);
void pzgemm_(const char *, const char *, int *, int *, const int *,
             std::complex<double> *, std::complex<double> *, int *, int *,
             const int *, std::complex<double> *, int *, int *, const int *,
             std::complex<double> *, std::complex<double> *, int *, int *,
             int *);
void pzheev_(char *, char *, int *, std::complex<double> *, int *, int *, int *,
             double *, std::complex<double> *, int *, int *, int *,
             std::complex<double> *, int *, std::complex<double> *, int *,
             int *);
void pdsygvx_(const int *, const char*, const char*, const char*, const int*,
                double*, int*, int*, int*, double*, int*, int*, int*, double*,
                double*, int*, int*, double*, int*, int*,
                double*, double*, double*, int*, int*, int*, double*,
                int*, int*, int*, int*, int*, double*, int*);
// calculate some eigenvalues of a parallel matrix of doubles
void pdsyevr_(const char*, const char*, const char*, const int*, double*, int*, int*, int*,
        double*, double*, int*, int*, int*, int*,
        double*, double*, int*, int*, int*,
        double*, int*, int*, int*, int*);
// calculate all eigenvalues and vectors by divide and conquer algorithm
void pdsyevd_(char *, char *, int *, double *, int *, int *, int *, double *,
             double *, int *, int *, int *, double *, int *, int *, int *, int *);
// take the transpose of a real matrix
void pdtran_(int * m, int * n, double * alpha, double * a, int * ia, int * ja, int * desc_a, double * beta, double * c, int * ic, int * jc, int * desc_c);

}

#endif /* BLACS_H */
