#include "phel_scattering_matrix.h"
#include "constants.h"
#include "io.h"
#include "mpiHelper.h"
#include "periodic_table.h"

// TODO el and get ph bandstructure getters are nice.
// Could we perhaps extend this to all the scattering matrix objects?

// TODO I think we should work out how to delete this class soon 

// this matrix class is just essentially a linewidths container,
// as in the phel linewidths + ph ph scattering thermal conductivity app,
// it's nice to separate and print out the two contributions.

PhElScatteringMatrix::PhElScatteringMatrix(Context &context_,
                                           StatisticsSweep &statisticsSweep_,
                                           BaseBandStructure &phBandStructure_,
                                           PhononH0 *phononH0_, 
                                           ElectronH0Wannier *electronH0_)
    : ScatteringMatrix(context_, statisticsSweep_, phBandStructure_, phBandStructure_),
     BasePhScatteringMatrix(context_, statisticsSweep_, phBandStructure_, phBandStructure_),
     phononH0(phononH0_), electronH0(electronH0_) {

  // this is true as the symmetrization isn't relevant here, this is
  // diagonal only and doesn't have any n^2 terms as the other phonon transport methods do
  isMatrixOmega = true;
  // automatically false, as phel scattering is only the diagonal
  highMemory = false;
}

// In the phononElectron case, we only compute the diagonal of the
// scattering matrix. Therefore, we compute only the linewidths
void PhElScatteringMatrix::builder(std::shared_ptr<VectorBTE> linewidth,
                                   [[maybe_unused]] std::vector<VectorBTE> &inPopulations,
                                   [[maybe_unused]] std::vector<VectorBTE> &outPopulations) {

  if (linewidth == nullptr) {
    Error("builderPhEl found a non-supported case");
  }
  if (linewidth->dimensionality != 1) {
    Error("Linewidths shouldn't have dimensionality");
  }

  // construct electronic band structure
  Points fullPoints(getPhBandStructure().getPoints().getCrystal(), context.getKMesh());
  auto t3 = ActiveBandStructure::builder(context, *electronH0, fullPoints);
  auto elBandStructure = std::get<0>(t3);
  // TODO this is super dangerous, it will work here but ! 
  // the fact that addPhElScattering cannot check if this is an el or ph statistics sweep is trap!
  statisticsSweep = std::get<1>(t3);

  // TODO we should let this go out of scope 
  // read in elph coupling 
  InteractionElPhWan couplingElPh = 
                      InteractionElPhWan::parse(context, 
                      innerBandStructure.getPoints().getCrystal(), // crystal associated with ph 
                      *phononH0);

  // compute the phonon electron lifetimes
  addPhElScattering(*this, context, getPhBandStructure(), elBandStructure,
                    statisticsSweep, 
                    couplingElPh, linewidth);

  // reduce as this is parallelized over mpi processes for wavevectrors
  mpi->allReduceSum(&linewidth->data);

  // Average over degenerate eigenstates.
  // we turn it off for now and leave the code if needed in the future
  degeneracyAveragingLinewidths(linewidth);

}
