#include "scattering_matrix.h"
#include "constants.h"
#include "io.h"
#include "mpiHelper.h"

// 3 cases:
// theMatrix and linewidth is passed: we compute and store in memory the
// scattering
//       matrix and the diagonal
// inPopulation+outPopulation is passed: we compute the action of the
//       scattering matrix on the in vector, returning outVec = sMatrix*vector
// only linewidth is passed: we compute only the linewidths

// BOUNDARY SCATTERING ==============================================
void addBoundaryScattering(ScatteringMatrix &matrix, Context &context,
                                std::vector<VectorBTE> &inPopulations,
                                std::vector<VectorBTE> &outPopulations,
                                int switchCase,
                                BaseBandStructure &outerBandStructure,
                                VectorBTE *linewidth) {

  if(mpi->mpiHead()) {
    std::cout <<
        "Adding boundary scattering to the scattering matrix." << std::endl;
  }

  double boundaryLength = context.getBoundaryLength();
  auto excludeIndices = matrix.excludeIndices;
  StatisticsSweep *statisticsSweep = &(matrix.statisticsSweep);
  int switchCase = matrix.switchCase;
  Particle particle = outerBandStructure.getParticle();
  int numCalculations = matrix.numCalculations;

  Kokkos::Profiling::pushRegion("boundary scattering");

  std::vector<int> is1s = outerBandStructure.irrStateIterator();
  int nis1s = is1s.size();

#pragma omp parallel for default(none) shared(                            \
  outerBandStructure, numCalculations, statisticsSweep, boundaryLength,   \
  particle, outPopulations, inPopulations, linewidth, switchCase, nis1s, is1s)
  for (int iis1 = 0; iis1 < nis1s; iis1++) {
    int is1 = is1s[iis1];
    StateIndex is1Idx(is1);
    auto vel = outerBandStructure.getGroupVelocity(is1Idx);
    double energy = outerBandStructure.getEnergy(is1Idx);
    int iBte1 = outerBandStructure.stateToBte(is1Idx).get();

    // this applies only to phonons
    if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte1) !=
        excludeIndices.end()) {
      continue;
    }

    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

      // for the case of phonons, we need to divide by a pop term.
      // might want to actually change this to "isMatrixOmega"
      // rather than isPhonon
      double rate = 0;

      if(particle.isPhonon()) {
        double temperature =
          statisticsSweep->getCalcStatistics(iCalc).temperature;
        double termPop = particle.getPopPopPm1(energy, temperature);
        rate = vel.squaredNorm() / boundaryLength * termPop;
      } else {
        rate = vel.squaredNorm() / boundaryLength;
      }

      if (switchCase == 0) {// case of matrix construction
        linewidth->operator()(iCalc, 0, iBte1) += rate;

      } else if (switchCase == 1) {// case of matrix-vector multiplication
        for (unsigned int iVec = 0; iVec < inPopulations.size(); iVec++) {
          for (int i = 0; i < 3; i++) {
            outPopulations[iVec](iCalc, i, iBte1) +=
                rate * inPopulations[iVec](iCalc, i, iBte1);
          }
        }

      } else {// case of linewidth construction
        linewidth->operator()(iCalc, 0, iBte1) += rate;
      }
    }
  }
  Kokkos::Profiling::popRegion();
}

