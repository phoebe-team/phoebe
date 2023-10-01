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
// Important NOTE: this is looped over parallel Irr states,
// so after using it, one must call allReduceSum on the linewidths,
// as is currently done in electron and phonon scattering matrixes
void addBoundaryScattering(ScatteringMatrix &matrix, Context &context,
                                std::vector<VectorBTE> &inPopulations,
                                std::vector<VectorBTE> &outPopulations,
                                int switchCase,
                                BaseBandStructure &bandStructure,
                                std::shared_ptr<VectorBTE> linewidth) {
  if(mpi->mpiHead()) {
    std::cout <<
        "Adding boundary scattering to the scattering matrix." << std::endl;
  }

  double boundaryLength = context.getBoundaryLength();
  auto excludeIndices = matrix.excludeIndices;
  StatisticsSweep *statisticsSweep = &(matrix.statisticsSweep);
  Particle particle = bandStructure.getParticle();
  int numCalculations = matrix.numCalculations;

  Kokkos::Profiling::pushRegion("boundary scattering");

  // loop over only parallel states, because later the linewidths will be allReduceSum'd
  std::vector<int> is1s = bandStructure.parallelIrrStateIterator();

  #pragma omp parallel for default(none) shared(                            \
   bandStructure, numCalculations, statisticsSweep, boundaryLength,   \
   particle, outPopulations, inPopulations, linewidth, switchCase, excludeIndices, is1s, matrix)
  for (int iis1 = 0; iis1 < is1s.size(); iis1++ ) {

      int is1 = is1s[iis1]; // we do this because an iterator cannot be omp parallelized

      StateIndex is1Idx(is1);
      auto vel = bandStructure.getGroupVelocity(is1Idx);
      int iBte1 = bandStructure.stateToBte(is1Idx).get();

      // shift the indices if it's necessary
      if(matrix.isCoupled) {
        std::tuple<int,int> tup =
               matrix.shiftToCoupledIndices(iBte1, iBte1, particle, particle);
        iBte1 = std::get<0>(tup); // this time we only need diagonal
      }

      // this applies only to phonons
      if (std::find(excludeIndices.begin(), excludeIndices.end(), iBte1) != excludeIndices.end()) {
        continue;
      }

      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

        double rate = 0;

        // for the case of phonons, we need to divide by a pop term.
        // TODO might want to actually change this to "isMatrixOmega" rather than isPhonon?
        if(particle.isPhonon()) {
          double temperature = statisticsSweep->getCalcStatistics(iCalc).temperature;
          double energy = bandStructure.getEnergy(is1Idx);
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

