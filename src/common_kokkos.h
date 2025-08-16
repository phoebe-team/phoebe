#ifndef COMMON_KOKKOS_H
#define COMMON_KOKKOS_H

#include <Kokkos_Core.hpp>

/** Define some useful classes for Kokkos-related calculations.
 */

// Kokkos View types for global memory spaces
using ComplexView1D = Kokkos::View<Kokkos::complex<double> *, Kokkos::LayoutRight>;
using ComplexView2D = Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight>;
using ComplexView3D = Kokkos::View<Kokkos::complex<double> ***, Kokkos::LayoutRight>;
using ComplexView4D = Kokkos::View<Kokkos::complex<double> ****, Kokkos::LayoutRight>;
using ComplexView5D = Kokkos::View<Kokkos::complex<double> *****, Kokkos::LayoutRight>;
using ComplexView6D = Kokkos::View<Kokkos::complex<double> ******, Kokkos::LayoutRight>;
using IntView1D = Kokkos::View<int *, Kokkos::LayoutRight>;
using IntView2D = Kokkos::View<int **, Kokkos::LayoutRight>;
using DoubleView1D = Kokkos::View<double *, Kokkos::LayoutRight>;
using DoubleView2D = Kokkos::View<double **, Kokkos::LayoutRight>;
using DoubleView3D = Kokkos::View<double ***, Kokkos::LayoutRight>;
using DoubleView4D = Kokkos::View<double ****, Kokkos::LayoutRight>;
using DoubleView5D = Kokkos::View<double *****, Kokkos::LayoutRight>;
using StridedComplexView3D = Kokkos::View<Kokkos::complex<double> ***, Kokkos::LayoutStride>;

using HostComplexView1D = Kokkos::View<Kokkos::complex<double> *, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostComplexView2D = Kokkos::View<Kokkos::complex<double> **, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostComplexView3D = Kokkos::View<Kokkos::complex<double> ***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostComplexView4D = Kokkos::View<Kokkos::complex<double> ****, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostComplexView5D = Kokkos::View<Kokkos::complex<double> *****, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostDoubleView1D = Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostDoubleView2D = Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostDoubleView3D = Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostDoubleView4D = Kokkos::View<double ****, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostDoubleView5D = Kokkos::View<double *****, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// Kokkos Range types
using Range2D = Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using Range4D = Kokkos::MDRangePolicy<Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using Range5D = Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using Range6D = Kokkos::MDRangePolicy<Kokkos::Rank<6, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;

/** This function does a batched diagonalization of M matrices A.
 * Each kernel works on a single matrix A
 *
 * @param A. On entry, a MxNxN tensor, identifying M hermitian matrices of size
 * NxN. matrix. On exit, A contains the eigenvectors of all the M matrices.
 * @param W: a MxN tensor, containing the N eigenvalues of each M-th matrix A.
 */
void kokkosZHEEV(StridedComplexView3D& A, DoubleView2D& W);

class DeviceManager {
 public:
  DeviceManager();
  /** Let the manager know that some memory on the device has been allocated.
   *
   * @param memoryBytes :amount of allocated memory in bytes.
   */
  void addDeviceMemoryUsage(const double& memoryBytes);

  /** let the manager know that some memory on the device has been freed.
   *
   * @param memoryBytes: amount of deallocated memory in Bytes
   */
  void removeDeviceMemoryUsage(const double& memoryBytes);

  /** Get how much memory is left on the kokkos device.
   *
   * @return memory left in bytes.
   */
  double getAvailableMemory();

  /** Returns the total memory present on the kokkos device.
   * This value is set by the user with the MAXMEM environment variable.
   *
   * @return device memory in bytes.
   */
  double getTotalMemory();

  /** splits a vector into batches of size batchSize.
   * Note: batchSize is adjusted, if necessary.
   * This function is used to launch a kokkos loop on the GPU. In Phoebe, the
   * memory grows linearly with the number of k-points in the interpolation.
   * So, this is an auxiliary utility to see how many k-points we can fit in the
   * device (GPU) memory before using it up completely.
   *
   * @param ikIterator: vector of integers that needs to be looped over.
   * @param batchSize: in input, it sets the largest size of the batch allowed.
   * Scaled down if batchSize is bigger than the ikIterator.size().
   * @return batches: a vector<vector<int>> which splits the original vector.
   */
  std::vector<std::vector<int>> splitToBatches(
      const std::vector<int>& ikIterator, int& batchSize);

 private:
  double memoryUsed = 0.;
  double memoryTotal = 0.;
};

// define a global object used for managing the memory on the GPU
extern DeviceManager* kokkosDeviceMemory;

// function to initialize the Kokkos environment
void initKokkos(int argc, char *argv[]);

// function to delete the Kokkos environment
void deleteKokkos();

// print info about the Device memory used by the device
void kokkosInfo();

// functions for printing Kokkos Views for debugging
template<class VT>
void print1D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  for(int i0 = 0; i0 < n0; i0++){
    printf("%s%.16g\n", prefix, h(i0));
  }
}
template<class VT>
void print1DComplex(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  for(int i0 = 0; i0 < n0; i0++){
    printf("%s%.16g %.16g\n", prefix, h(i0).real(), h(i0).imag());
  }
}
template<class VT>
void print2D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      printf("%s%.16g\n", prefix, h(i0,i1));
    }
  }
}
template<class VT>
void print2DComplex(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      printf("%s%.16g %.16g\n", prefix, h(i0,i1).real(), h(i0,i1).imag());
    }
  }
}
template<class VT>
void print3D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        printf("%s%.16g\n", prefix, h(i0,i1,i2));
      }
    }
  }
}
template<class VT>
void print3DComplex(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        printf("%s%.16g %.16g\n", prefix, h(i0,i1,i2).real(), h(i0,i1,i2).imag());
      }
    }
  }
}
template<class VT>
void print4D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  int n3 = h.extent(3);
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        for(int i3 = 0; i3 < n3; i3++){
          printf("%s%.16g\n", prefix, h(i0,i1,i2,i3));
        }
      }
    }
  }
}
template<class VT>
void print4DComplex(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  int n3 = h.extent(3);
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        for(int i3 = 0; i3 < n3; i3++){
          printf("%s%.16g %.16g\n", prefix, h(i0,i1,i2,i3).real(), h(i0,i1,i2,i3).imag());
        }
      }
    }
  }
}

template<class VT>
void printNorm1D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  double norm = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    printf("%s%.16g\n", prefix, h(i0));
    norm += h(i0)*h(i0);
  }
  printf("%s%.16e\n", prefix, norm);
}
template<class VT>
void printNorm1DComplex(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  double norm = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    norm += h(i0).real()*h(i0).real() + h(i0).imag()*h(i0).imag();
  }
  printf("%s%.16e\n", prefix, norm);
}
template<class VT>
void printNorm2D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  double norm = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      norm += h(i0,i1)*h(i0,i1);
    }
  }
  printf("%s%.16e\n", prefix, norm);
}
template<class VT>
void printNorm2DComplex(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  double norm = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      norm += h(i0,i1).real()*h(i0,i1).real() + h(i0,i1).imag()*h(i0,i1).imag();
    }
  }
  printf("%s%.16e\n", prefix, norm);
}
template<class VT>
void printNorm3D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  double norm = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        norm += h(i0,i1,i2)*h(i0,i1,i2);
      }
    }
  }
  printf("%s%.16e\n", prefix, norm);
}
template<class VT>
void printNorm3DComplex(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  double norm = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        norm += h(i0,i1,i2).real()*h(i0,i1,i2).real() + h(i0,i1,i2).imag()*h(i0,i1,i2).imag();
      }
    }
  }
  printf("%s%.16e\n", prefix, norm);
}
template<class VT>
void printSum4D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  int n3 = h.extent(3);
  double sum = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        for(int i3 = 0; i3 < n3; i3++){
          sum += h(i0,i1,i2,i3);
        }
      }
    }
  }
  printf("%s%.16e\n", prefix, sum);
}
template<class VT>
void printNorm4D(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  int n3 = h.extent(3);
  double norm = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        for(int i3 = 0; i3 < n3; i3++){
          norm += h(i0,i1,i2,i3)*h(i0,i1,i2,i3);
        }
      }
    }
  }
  printf("%s%.16e\n", prefix, norm);
}
template<class VT>
void printNorm4DComplex(const char* prefix, VT d){
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);

  int n0 = h.extent(0);
  int n1 = h.extent(1);
  int n2 = h.extent(2);
  int n3 = h.extent(3);
  double norm = 0.0;
  for(int i0 = 0; i0 < n0; i0++){
    for(int i1 = 0; i1 < n1; i1++){
      for(int i2 = 0; i2 < n2; i2++){
        for(int i3 = 0; i3 < n3; i3++){
          norm += h(i0,i1,i2,i3).real()*h(i0,i1,i2,i3).real() + h(i0,i1,i2,i3).imag()*h(i0,i1,i2,i3).imag();
        }
      }
    }
  }
  printf("%s%.16e\n", prefix, norm);
}

#endif
