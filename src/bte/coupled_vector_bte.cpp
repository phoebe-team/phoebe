#include "coupled_vector_bte.h"
#include "constants.h"

// TODO what should I do about stat sweep?
// Can we enure it's an el one? I think they're the same except for mu.
// TODO can the vector bte get it's own "getIrrStateIterator" that
// would replace the bandstructure's one? Then VBTE can return the
// bandstructure's distribution and this one can do it's own thing.

/* NOTE: for now this object is basically a simple container with indexing.
 * The relaxons solver doesn't have much need for the vectorBTE object,
 * but we want to keep a copy of the linewidths for testing purposes/scattering
 * rate calculations.
 */

// default constructor
CoupledVectorBTE::CoupledVectorBTE(StatisticsSweep &statisticsSweep_,
                     BaseBandStructure &phBandStructure_,
                     BaseBandStructure &elBandStructure_,
                     const int &dimensionality_)
    : VectorBTE(statisticsSweep_, elBandStructure_, dimensionality_),
       phBandStructure(phBandStructure_) {

  if (dimensionality_ <= 0) {
    Error("BaseVectorBTE doesn't accept <=0 dimensions");
  }
  // check that the bandstructures are right
  if (bandStructure.getParticle().isPhonon() || !phBandStructure.getParticle().isPhonon()) {
    Error("Developer error: You've tried to create a coupled VBTE"
        " object with the el and ph bandstructures flipped!");
  }
  // TODO enforce that this is a stat sweep assocaited with electrons?

  dimensionality = dimensionality_;
  // num states here is electron first, then phonon in size.
  // dimensions of the whole vector are nElStates, nPhStates
  numStates = int(bandStructure.irrStateIterator().size()) +
                        int(phBandStructure.irrStateIterator().size());
  numCalculations = statisticsSweep.getNumCalculations();
  numCalculations *= dimensionality;

  numChemPots = statisticsSweep.getNumChemicalPotentials();
  numTemps = statisticsSweep.getNumTemperatures();
  data.resize(numCalculations, numStates);
  data.setZero();

  // TODO can we abstract out a helper function for getExcludedIndices?
  for (int is : phBandStructure.irrStateIterator()) {
    auto isIdx = StateIndex(is);
    double en = phBandStructure.getEnergy(isIdx);
    if (en < phEnergyCutoff) { // cutoff at 0.1 cm^-1
      int iBte = phBandStructure.stateToBte(isIdx).get();
      excludeIndices.push_back(iBte);
    }
  }
}

// TODO try removing the old VBTE ones
// copy constructor
//CoupledVectorBTE::CoupledVectorBTE(const VectorBTE &that) : VectorBTE(that) { //,
  // this copies everything except the phonon band structure...
//}

// TODO try removing the old VBTE ones
// copy constructor
//CoupledVectorBTE::CoupledVectorBTE(const CoupledVectorBTE &that) : VectorBTE(that),
//                phBandStructure(that.phBandStructure) {
//}

/*
// copy assignment
CoupledVectorBTE &CoupledVectorBTE::operator=(const CoupledVectorBTE &that) {
  if (this != &that) {
    phBandStructure = that.phBandStructure;
    statisticsSweep = that.statisticsSweep;
    bandStructure = that.bandStructure;
    numCalculations = that.numCalculations;
    numStates = that.numStates;
    numChemPots = that.numChemPots;
    numTemps = that.numTemps;
    dimensionality = that.dimensionality;
    data = that.data;
    excludeIndices = that.excludeIndices;
  }
  return *this;
}
*/

// product operator overload
Eigen::MatrixXd CoupledVectorBTE::dot([[ maybe_unused ]] const CoupledVectorBTE &that) {

  Error("Developer error: for now, CVBTE dot function is not implemented.");
  Eigen::MatrixXd dummy; // to quiet warnings
  return dummy;

}

// this should affect all the operators
CoupledVectorBTE CoupledVectorBTE::baseOperator( [[maybe_unused]] CoupledVectorBTE &that, [[maybe_unused]] const int &operatorType) {
  Error("Developer error: for now, CVBTE operator function are not implemented.");
  return *this; // silence warnings
}

void CoupledVectorBTE::population2Canonical() {
  Error("Developer error: for now, CVBTE pop2Canonical function is not implemented.");
}
void CoupledVectorBTE::canonical2Population() {
  Error("Developer error: for now, CVBTE pop2Canonical function is not implemented.");
}
// negation operator overload
CoupledVectorBTE CoupledVectorBTE::operator-() {
  CoupledVectorBTE newPopulation(statisticsSweep, phBandStructure, bandStructure, dimensionality);
  newPopulation.data = -this->data;
  return newPopulation;
}

CoupledVectorBTE CoupledVectorBTE::sqrt() {
  CoupledVectorBTE newPopulation(statisticsSweep, phBandStructure, bandStructure, dimensionality);
  newPopulation.data << this->data.array().sqrt();
  return newPopulation;
}

CoupledVectorBTE CoupledVectorBTE::reciprocal() {
  CoupledVectorBTE newPopulation(statisticsSweep, phBandStructure, bandStructure, dimensionality);
  #pragma omp parallel for
  for (int iBte = 0; iBte < numStates; iBte++) {
    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      for (int iDim = 0; iDim < dimensionality; iDim++) {
        // if the linewidth is somehow zero, we should leave the recip value as
        // zero so that we don't count these states.
        if( CoupledVectorBTE::operator()(iCalc, iDim, iBte) != 0.) {
          newPopulation(iCalc, iDim, iBte) = 1./CoupledVectorBTE::operator()(iCalc, iDim, iBte);
        }
      }
    }
  }
  return newPopulation;
}

void CoupledVectorBTE::setConst([[maybe_unused]] const double &constant) {
  Error("Developer error: Does not make sense to set cBTE to one constant value.");
}

