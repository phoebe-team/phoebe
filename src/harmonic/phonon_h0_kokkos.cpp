#include "phonon_h0.h"

/**
 * Build the Hamiltonian matrix for a batch of k-points
 */
StridedComplexView3D PhononH0::kokkosBatchedBuildBlochHamiltonian(
    const DoubleView2D &cartesianCoordinates) {

  int numK = cartesianCoordinates.extent(0);
  int numBands = this->numBands;
  int numBravaisVectors = this->numBravaisVectors;
  Kokkos::complex<double> complexI(0.0, 1.0);

  auto bravaisVectors_d = this->bravaisVectors_d;
  auto weights_d = this->weights_d;
  auto mat2R_d = this->mat2R_d;
  auto numAtoms = this->numAtoms;

  ComplexView3D dynamicalMatrices("dynMat", numK, numBands, numBands);

  // Compute all dot products of k points (phases)
  ComplexView2D phases_d("phPhases_d", numK, numBravaisVectors);
  Kokkos::parallel_for(
      "el_phases", Range2D({0, 0}, {numK, numBravaisVectors}),
      KOKKOS_LAMBDA(int iK, int iR) {
        double arg = 0.0;
        for (int i = 0; i < 3; i++) {
          arg += cartesianCoordinates(iK, i) * bravaisVectors_d(iR, i);
        }
        phases_d(iK, iR) = exp(-complexI * arg) * weights_d(iR);
      });
  Kokkos::fence();
  //print2DComplex("new = ", phases_d);
  //print2D("new = ", bravaisVectors_d);
  //print3D("new = ", mat2R_d);

  // multiply matrix by phase
  Kokkos::parallel_for(
      "el_hamilton", Range3D({0, 0, 0}, {numK, numBands, numBands}),
      KOKKOS_LAMBDA(int iK, int m, int n) {
        Kokkos::complex<double> tmp(0.0);
        for (int iR = 0; iR < numBravaisVectors; iR++) {
          tmp += phases_d(iK, iR) * mat2R_d(m, n, iR);
        }
        dynamicalMatrices(iK, m, n) = tmp;
      });
  Kokkos::realloc(phases_d, 0, 0);

  if (hasDielectric) {
    auto longRangeCorrection1_d = this->longRangeCorrection1_d;

    // add constant long range term
    Kokkos::parallel_for(
        "el_hamilton", Range3D({0, 0, 0}, {numK, 3, 3}),
        KOKKOS_LAMBDA(int iK, int i, int j) {
          auto D = Kokkos::subview(dynamicalMatrices, iK, Kokkos::ALL, Kokkos::ALL);
          for (int iAt=0; iAt<numAtoms; ++iAt) {
            int m = iAt * 3 + i;
            int n = iAt * 3 + j;
            D(m, n) += longRangeCorrection1_d(i, j, iAt);
          }
        });
    Kokkos::fence();

    double norm = e2 * fourPi / volumeUnitCell;
    int numG = gVectors_d.extent(0);

    auto gVectors_d = this->gVectors_d;
    auto dielectricMatrix_d = this->dielectricMatrix_d;
    auto bornCharges_d = this->bornCharges_d;
    auto atomicPositions_d = this->atomicPositions_d;
    auto gMax = this->gMax;

    // per-thread scratch memory size
    int byte_size = Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::scratch_memory_space>::shmem_size(numBands);

    // compute full long range term
    Kokkos::parallel_for("long-range-ph-H0",
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(numG, Kokkos::AUTO()).set_scratch_size(
          0, Kokkos::PerThread(byte_size)),
        KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type &team) {

          int iG = team.league_rank();
          auto G = Kokkos::subview(gVectors_d, iG, Kokkos::ALL);

          // allocate per-thread scratch memory
          Kokkos::View<double*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::scratch_memory_space> GQZ(team.thread_scratch(0), 3*numAtoms);

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, numK), [&] (const int iK){

            auto Q = Kokkos::subview(cartesianCoordinates, iK, Kokkos::ALL);
            auto D = Kokkos::subview(dynamicalMatrices, iK, Kokkos::ALL, Kokkos::ALL);

            double GQ[3];
            for (int i=0; i<3; ++i) {
              GQ[i] = G(i) + Q(i);
            }

            double geg = 0.;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, 9), [&] (const int ij, double &geg){
                int i = ij / 3;
                int j = ij % 3;
                geg += GQ[i] * dielectricMatrix_d(i, j) * GQ[j];
            }, geg);

            if (geg > 0. && geg < 4. * gMax) {
              double normG = norm * exp(-geg * 0.25) / geg;

              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 3*numAtoms), [&] (const int inb){
                  int nb = inb / 3;
                  int i = inb % 3;
                  double tmp = 0.0;
                  for (int j : {0, 1, 2}) {
                     tmp += GQ[j] * bornCharges_d(i, j, nb);
                  }
                  GQZ[inb] = tmp;
              });

              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, numAtoms*numAtoms), [&] (const int nanb){
                  int na = nanb / numAtoms;
                  int nb = nanb % numAtoms;
                  double arg = 0.;
                  for (int i = 0; i < 3; i++) {
                    double xi = atomicPositions_d(na,i) - atomicPositions_d(nb,i);
                    arg += xi * GQ[i];
                  }
                  Kokkos::complex<double> phase = normG * exp(complexI * arg);
                  for (int j : {0, 1, 2}) {
                    for (int i : {0, 1, 2}) {
                      Kokkos::complex<double> x = phase * GQZ[na*3+i] * GQZ[nb*3+j];
                      Kokkos::atomic_add(&D(na*3+i, nb*3+j), x);
                      // there's a reduce on iG!
                    }
                  }
              });
            }
          });
        });
  }


  // Col-major matrices stored contiguously
  Kokkos::LayoutStride DDlayout(
      numK, numBands*numBands,
      numBands, 1,
      numBands, numBands
  );

  // ensure hermiticity
  StridedComplexView3D DD("dynMat", DDlayout);
  Kokkos::parallel_for(
      "el_hamilton", Range3D({0, 0, 0}, {numK, numBands, numBands}),
      KOKKOS_LAMBDA(int iK, int m, int n) {
        auto D = Kokkos::subview(dynamicalMatrices, iK, Kokkos::ALL, Kokkos::ALL);
        DD(iK, m, n) = 0.5 * ( D(m,n) + Kokkos::conj(D(n,m)) );
      });
  Kokkos::realloc(dynamicalMatrices, 0, 0, 0);

  auto atomicMasses_d = this->atomicMasses_d;

  // divide by atomic masses
  Kokkos::parallel_for(
      "el_hamilton", Range3D({0, 0, 0}, {numK, numBands, numBands}),
      KOKKOS_LAMBDA(int iK, int m, int n) {
        DD(iK, m, n) /= sqrt(atomicMasses_d(m) * atomicMasses_d(n));
      });

  return DD;
}

/**
 * Create and diagonalize Hamiltonians for a batch of k-points
 */
std::tuple<DoubleView2D, StridedComplexView3D> PhononH0::kokkosBatchedDiagonalizeFromCoordinates(
    const DoubleView2D &cartesianCoordinates, const bool withMassScaling) {

  // create the Hamiltonians
  StridedComplexView3D dynamicalMatrices =
      kokkosBatchedBuildBlochHamiltonian(cartesianCoordinates);

  int numK = dynamicalMatrices.extent(0);
  int numBands = this->numBands;
  DoubleView2D frequencies("energies", numK, numBands);

  // perform diagonalization of all matrices
  kokkosZHEEV(dynamicalMatrices, frequencies);

  // frequencies are sign(eigenvalues)*sqrt(abs(eigenvalues))
  Kokkos::parallel_for(
      "el_hamilton", Range2D({0, 0}, {numK, numBands}),
      KOKKOS_LAMBDA(int iK, int m) {
        if (frequencies(iK, m) > 0.) {
          frequencies(iK, m) = sqrt(frequencies(iK, m));
        } else {
          frequencies(iK, m) = -sqrt(-frequencies(iK, m));
        }
      });

  if(withMassScaling) kokkosBatchedScaleEigenvectors(dynamicalMatrices);
  return std::make_tuple(frequencies, dynamicalMatrices);
}

void PhononH0::kokkosBatchedScaleEigenvectors(StridedComplexView3D& eigenvectors) {
  int numK = eigenvectors.extent(0);
  int numBands = this->numBands;
  auto atomicMasses_d = this->atomicMasses_d;

  Kokkos::parallel_for(
      "mass_rescale", Range3D({0, 0, 0}, {numK, numBands, numBands}),
      KOKKOS_LAMBDA(int iK, int m, int n) {
        eigenvectors(iK, m, n) /= sqrt(atomicMasses_d(m));
      });
}

double PhononH0::getDeviceMemoryUsage() {
  double memory = 16 * longRangeCorrection1_d.size()
      + 8 * (atomicMasses_d.size() + gVectors_d.size()
           + dielectricMatrix_d.size() + bornCharges_d.size()
           + atomicPositions_d.size() + bravaisVectors_d.size()
           + weights_d.size() + mat2R_d.size() );
  return memory;
}

int PhononH0::estimateBatchSize(const bool& withVelocity) {
  double memoryAvailable = kokkosDeviceMemory->getAvailableMemory();

  int matrixSize = numBands * numBands;
  int transformSize = matrixSize + numBravaisVectors;
  transformSize = std::max(transformSize, 2*matrixSize + numBands);

  double memoryPerPoint = 0.;
  if (withVelocity) {
    memoryPerPoint += 16. * (transformSize*7 + 2*matrixSize + matrixSize*3);
  } else {
    memoryPerPoint += 16. * transformSize;
  }
  //printf("memory per point = %g, available = %g\n", memoryPerPoint, memoryAvailable);

  // we try to use 90% of the available memory (leave some buffer)
  int numBatches = int(memoryAvailable / memoryPerPoint * 0.95);
  if (numBatches < 1) {
    Error("Not enough memory available on device, try reduce memory usage");
  }

  return numBatches;
}

/**
 * Make a bandstructure for all the k-points
 */
FullBandStructure PhononH0::kokkosPopulate(Points &fullPoints,
                                           const bool &withVelocities,
                                           const bool &withEigenvectors,
                                           const bool isDistributed) {

  int numBands = getNumBands();
  auto particle = getParticle();

  FullBandStructure fullBandStructure(numBands, particle, withVelocities,
                                      withEigenvectors, fullPoints,
                                      isDistributed);

  // divide the k-points into batches
  std::vector<std::vector<int>> ikBatches;
  {
    std::vector<int> ikIterator = fullBandStructure.getWavevectorIndices();
    int batchSize = estimateBatchSize(withVelocities);
    ikBatches = kokkosDeviceMemory->splitToBatches(ikIterator, batchSize);
  }

  // loop over the batches
  for (auto ikBatch : ikBatches) {

    int numK = ikBatch.size();

    // get all the k-points in a batch
    DoubleView2D cartesianWavevectors_d("el_cartWav_d", numK, 3);
    {
      auto cartesianWavevectors_h = Kokkos::create_mirror_view(cartesianWavevectors_d);
#pragma omp parallel for
      for (int iik = 0; iik < numK; ++iik) {
        int ik = ikBatch[iik];
        WavevectorIndex ikIdx(ik);
        Eigen::Vector3d k = fullBandStructure.getWavevector(ikIdx);
        for (int i = 0; i < 3; ++i) {
          cartesianWavevectors_h(iik, i) = k(i);
        }
      }
      Kokkos::deep_copy(cartesianWavevectors_d, cartesianWavevectors_h);
    }

    // do all diagonalizations at once with Kokkos
    {
      DoubleView2D allEnergies_d;
      StridedComplexView3D allEigenvectors_d;
      ComplexView4D allVelocities_d;
      if (withVelocities) {
        auto t = kokkosBatchedDiagonalizeWithVelocities(cartesianWavevectors_d);
        allEnergies_d = std::get<0>(t);
        allEigenvectors_d = std::get<1>(t);
        allVelocities_d = std::get<2>(t);
      } else {
        auto t = kokkosBatchedDiagonalizeFromCoordinates(cartesianWavevectors_d);
        allEnergies_d = std::get<0>(t);
        allEigenvectors_d = std::get<1>(t);
      }
      Kokkos::realloc(cartesianWavevectors_d, 0, 0);


      // copy results to CPU
      auto allEnergies_h = Kokkos::create_mirror_view(allEnergies_d);
      auto allEigenvectors_h = Kokkos::create_mirror_view(allEigenvectors_d);
      Kokkos::deep_copy(allEnergies_h, allEnergies_d);
      Kokkos::deep_copy(allEigenvectors_h, allEigenvectors_d);

      // store the results in the old datastructures
#pragma omp parallel for
      for (int iik = 0; iik < numK; iik++) {
        int ik = ikBatch[iik];
        Point point = fullBandStructure.getPoint(ik);
        Eigen::VectorXd ens(numBands);
        for (int ib = 0; ib < numBands; ++ib) {
          ens(ib) = allEnergies_h(iik, ib);
        }
        fullBandStructure.setEnergies(point, ens);
        if (withEigenvectors) {
          Eigen::MatrixXcd eigVec(numBands, numBands);
          for (int ib1 = 0; ib1 < numBands; ++ib1) {
            for (int ib2 = 0; ib2 < numBands; ++ib2) {
              eigVec(ib1, ib2) = allEigenvectors_h(iik, ib1, ib2);
            }
          }
          fullBandStructure.setEigenvectors(point, eigVec);
        }
      }

      if (withVelocities) {
        auto allVelocities_h = Kokkos::create_mirror_view(allVelocities_d);
        Kokkos::deep_copy(allVelocities_h, allVelocities_d);
        for (int iik = 0; iik < numK; iik++) {
          int ik = ikBatch[iik];
          Point point = fullBandStructure.getPoint(ik);
          Eigen::Tensor<std::complex<double>,3> v(numBands, numBands, 3);
          for (int ib1 = 0; ib1 < numBands; ++ib1) {
            for (int ib2 = 0; ib2 < numBands; ++ib2) {
              for (int i = 0; i < 3; ++i) {
                v(ib1, ib2, i) = allVelocities_h(iik, ib1, ib2, i);
              }
            }
          }
          fullBandStructure.setVelocities(point, v);
        }
      }
      Kokkos::realloc(allEnergies_d, 0, 0);
      allEigenvectors_d = decltype(allEigenvectors_d)();
      Kokkos::realloc(allVelocities_d, 0, 0, 0, 0);
    }
  }

  return fullBandStructure;
}


/**
 * Create and diagonalize Hamiltonians for a batch of k-points, with velocities
 * Returns the energies (nk, nb), eigenvectors (nk, nb, nb)
 * and velocities (nk, nb, nb, 3) at each k-point.
 * Note that for each k-point, the (nb, nb) eigenvector
 * matrix is column-major, as required by the cuSOLVER routine,
 * hence the StridedLayout.
 */
std::tuple<DoubleView2D, StridedComplexView3D, ComplexView4D>
PhononH0::kokkosBatchedDiagonalizeWithVelocities(
    const DoubleView2D &cartesianCoordinates) {

  // Note: this is slightly different than electronH0Wannier
  // here, we need to compute the derivative of sqrt(DynamicalMatrix)
  // while for electrons we derive the BlochHamiltonian directly

  int numK = cartesianCoordinates.extent(0);

  double delta = 1.0e-8;
  Kokkos::complex<double> complexI(0.0, 1.0);

  // prepare all the wavevectors at which we need the hamiltonian,
  // (kx, ky, kz), (kx+dk, ky, kz), (kx - dk, ky, kz), (kx, ky + dk, kz), etc.
  // 7 here is for the seven possible k+/- values
  DoubleView2D allVectors("wavevectors_k", numK * 7, 3);
  Kokkos::parallel_for(
      "phHamiltonianShifted_d", numK, KOKKOS_LAMBDA(int iK) {
        // central points, i = x,y,z
        for (int i = 0; i < 3; ++i) {
          allVectors(iK * 7, i) = cartesianCoordinates(iK, i);
        }
        // iDir for each kx, ky, kz +/-, i = x,y,z
        for (int iDir = 0; iDir < 3; ++iDir) {
          for (int i = 0; i < 3; ++i) {
            allVectors(iK * 7 + iDir * 2 + 1, i) = cartesianCoordinates(iK, i);
            allVectors(iK * 7 + iDir * 2 + 2, i) = cartesianCoordinates(iK, i);
            if (iDir == i) {
              allVectors(iK * 7 + iDir * 2 + 1, i) += delta;
              allVectors(iK * 7 + iDir * 2 + 2, i) -= delta;
            }
          }
        }
      });

  // compute the electronic properties at all wavevectors
  auto t = kokkosBatchedDiagonalizeFromCoordinates(allVectors, false);
  DoubleView2D allEnergies = std::get<0>(t);
  StridedComplexView3D allEigenvectors = std::get<1>(t);

  int numBands = allEnergies.extent(1);

  Kokkos::LayoutStride eigenvectorLayout(
      numK, numBands*numBands,
      numBands, 1,
      numBands, numBands);

  StridedComplexView3D resultEigenvectors("eigenvectors", eigenvectorLayout);
  ComplexView4D resultVelocities("velocities", numK, numBands, numBands, 3);
  ComplexView2D phases("wallacePhases", numK, numBands);

  // copy slice of eigenvectors, energies into views 
  Kokkos::parallel_for(
      "phase", Range3D({0, 0, 0}, {numK, numAtoms, 3}), KOKKOS_LAMBDA(int iK, int iat, int i) {
        // prepare the Wallace phase 
          int m = i + iat * 3;  // generate the band index 
          auto R = atomicPositions.row(iat); // TODO need atomic positions as a view... 

          // i for each kx, ky, kz +/-, j = x,y,z
          double arg = 0.0;
          for(int j = 0; j < 3; ++j) {
            arg += R(j) * allVectors(iK * 7 + i * 2 + 1, j); 
          }
          phases(iK,m) = exp( -complexI * arg );
        });
  // apply phases -- here we use the results eigenvectors container 
  // to hold phase * U(k), to be replaced with standard ones before return 
  Kokkos::parallel_for(
      "der", Range2D({0, 0}, {numK, numBands}), KOKKOS_LAMBDA(int iK, int m) {

        auto X = Kokkos::subview(allEigenvectors, iK * 7, Kokkos::ALL, Kokkos::ALL);

        // copy resultEigenvectors to return at the end
        for (int n=0; n<numBands; ++n) {
          resultEigenvectors(iK, m, n) = phases(iK, m) * X(m, n);
        }
      });

  Kokkos::resize(phases, 0, 0); // deallocate

  // Views for intermediate results
  ComplexView3D der("der", numK, numBands, numBands);
  ComplexView3D tmpV("tmpV", numK, numBands, numBands);

  // kx, ky, kz directions 
  for (int i = 0; i < 3; ++i) {

    // TODO will these go out of scope? 
    ComplexView2D phasesPlus("wallacePhasesPlus", numK, numBands);
    ComplexView2D phasesMinus("wallacePhasesMinus", numK, numBands);

    // precalc +/- phases 
    Kokkos::parallel_for(
        "phasePlusMinus", Range3D({0, 0, 0}, {numK, numAtoms, 3}), KOKKOS_LAMBDA(int iK, int iat, int i) {
          // prepare the Wallace phase 

          int m = i + iat * 3;  // generate the band index 
          auto R = atomicPositions.row(iat); // TODO need atomic positions as a view... 

          // i for each kx, ky, kz +/-, j = x,y,z
          double argPlus = 0.0;
          double argMinus = 0.0;
          for(int j = 0; j < 3; ++j) {
            argPlus += R(j) * allVectors(iK * 7 + i * 2 + 1, j); 
            argMinus += R(j) * allVectors(iK * 7 + i * 2 + 2, j); 
          }
          phasesPlus(iK,m) = exp( -complexI * argPlus );
          phasesMinus(iK,m) = exp( -complexI * argMinus );
        });

    // To build the velocity operator, we first compute the matrix
    // der = dH/dk = ( H(k+dk)-H(k-dk) ) / 2dk
    // we also reinforce the Hermiticity (better numerical properties)
    Kokkos::parallel_for(
        "der", Range3D({0, 0, 0}, {numK, numBands, numBands}),
        KOKKOS_LAMBDA(int iK, int m, int n) {

          auto XPlus = Kokkos::subview(allEigenvectors, iK * 7 + i * 2 + 1, Kokkos::ALL, Kokkos::ALL);
          auto XMins = Kokkos::subview(allEigenvectors, iK * 7 + i * 2 + 2, Kokkos::ALL, Kokkos::ALL);

          DoubleView1D EPlus = Kokkos::subview(allEnergies, iK * 7 + i * 2 + 1, Kokkos::ALL);
          DoubleView1D EMins = Kokkos::subview(allEnergies, iK * 7 + i * 2 + 2, Kokkos::ALL);

          Kokkos::complex<double> x(0.,0.);
          for (int l=0; l<numBands; ++l) {
            // TODO check that these dimensions are right for band indices 
            x += phasesPlus(iK,m) * XPlus(m,l) * EPlus(l) * Kokkos::conj(phasesPlus(iK,n) * XPlus(n,l))
                -  phasesMinus(iK,m) * XMins(m,l) * EMins(l) * Kokkos::conj(phasesMinus(iK,n) * XMins(n,l));
          }
          der(iK, m, n) = x/(2*delta);
        });
    Kokkos::fence();

    Kokkos::resize(phasesPlus, 0, 0); // deallocate
    Kokkos::resize(phasesMinus, 0, 0); // deallocate

    // Now we complete the Hellman-Feynman theorem
    // and compute the velocity as v = U(k)^* der * U(k)

    // first applying v1 = U(k)^* der
    Kokkos::parallel_for(
        "tmpV", Range3D({0, 0, 0}, {numK, numBands, numBands}),
        KOKKOS_LAMBDA(int iK, int m, int n) {
          auto U = Kokkos::subview(resultEigenvectors, iK, Kokkos::ALL, Kokkos::ALL);
          auto dER = Kokkos::subview(der, iK, Kokkos::ALL, Kokkos::ALL);
          auto tempV = Kokkos::subview(tmpV, iK, Kokkos::ALL, Kokkos::ALL);
          Kokkos::complex<double> tmp(0.,0.);
          for (int l = 0; l < numBands; ++l) {
            // average (der^* + der / 2) to enforce Hermiticity 
            //tmp += Kokkos::conj(L(l,m)) * 0.5 * (R(l, n) + Kokkos::conj(R(n, l)));
            tmp += Kokkos::conj(U(l,m)) * dER(l, n);
          }
          tempV(m, n) = tmp;
        });
    Kokkos::fence();

    // then applying v_final = v1 * U(k)
    Kokkos::parallel_for(
        "vel", Range3D({0, 0, 0}, {numK, numBands, numBands}),
        KOKKOS_LAMBDA(int iK, int m, int n) {
          double norm = 0.;
          // TODO why skip gamma here but not above?
          for (int i=0; i<3; ++i) {
            norm += cartesianCoordinates(iK,i) * cartesianCoordinates(iK,i);
          }
          Kokkos::complex<double> tmp(0.,0.);
          if ( norm > 1.0e-6 ) {// skip the gamma point
            for (int l = 0; l < numBands; ++l) {
              tmp += tmpV(iK, m, l) * resultEigenvectors(iK, l, n);
            }
          }
          resultVelocities(iK, m, n, i) = tmp;
        });
  }

  // deallocate the scratch
  Kokkos::resize(der, 0, 0, 0);
  Kokkos::resize(tmpV, 0, 0, 0);

   // copy resultEigenvectors and energies to return 
  DoubleView2D resultEnergies("energies", numK, numBands);
  Kokkos::parallel_for(
      "copyEigenvectors", Range2D({0, 0}, {numK, numBands}), KOKKOS_LAMBDA(int iK, int m) {
        auto E = Kokkos::subview(allEnergies, iK * 7, Kokkos::ALL);
        auto X = Kokkos::subview(allEigenvectors, iK * 7, Kokkos::ALL, Kokkos::ALL);

       resultEnergies(iK, m) = E(m);

        for (int n=0; n<numBands; ++n) {
          resultEigenvectors(iK, m, n) = X(m, n);
        }
  });

  // deal with velocity issues and degenerate bands
  kokkosBatchedTreatDegenerateVelocities(cartesianCoordinates, resultEnergies,
                                         resultVelocities, 0.0001 / ryToCmm1);

// TODO: this is the only difference from the ElectronH0 kokkosPopulate()
// can we unify the two?
  kokkosBatchedScaleEigenvectors(resultEigenvectors);
  return std::make_tuple(resultEnergies, resultEigenvectors, resultVelocities);
}

