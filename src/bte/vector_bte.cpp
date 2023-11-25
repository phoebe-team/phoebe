#include "vector_bte.h"
#include "constants.h"

// default constructor
VectorBTE::VectorBTE(StatisticsSweep &statisticsSweep_,
                     BaseBandStructure &bandStructure_,
                     const int &dimensionality_)
    : statisticsSweep(statisticsSweep_), bandStructure(bandStructure_) {

  if (dimensionality_ <= 0) {
    Error("BaseVectorBTE doesn't accept <=0 dimensions");
  }

  dimensionality = dimensionality_;
  numStates = int(bandStructure.irrStateIterator().size());

  numCalculations = statisticsSweep.getNumCalculations();
  numCalculations *= dimensionality;

  numChemPots = statisticsSweep.getNumChemicalPotentials();
  numTemps = statisticsSweep.getNumTemperatures();
  data.resize(numCalculations, numStates);
  data.setZero();

  if (bandStructure.getParticle().isPhonon()) {
    for (int is : bandStructure.irrStateIterator()) {
      auto isIdx = StateIndex(is);
      double en = bandStructure.getEnergy(isIdx);
      // TODO standardize this number!
      if (en < 0.1 / ryToCmm1) { // cutoff at 0.1 cm^-1
        int iBte = bandStructure.stateToBte(isIdx).get();
        excludeIndices.push_back(iBte);
      }
    }
  }
}

// copy constructor
VectorBTE::VectorBTE(const VectorBTE &that) : statisticsSweep(that.statisticsSweep),
                                              bandStructure(that.bandStructure) {
  numCalculations = that.numCalculations;
  numStates = that.numStates;
  numChemPots = that.numChemPots;
  numTemps = that.numTemps;
  dimensionality = that.dimensionality;
  data = that.data;
  excludeIndices = that.excludeIndices;
}

// copy assignment
VectorBTE &VectorBTE::operator=(const VectorBTE &that) {
  if (this != &that) {
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

// product operator overload
Eigen::MatrixXd VectorBTE::dot(const VectorBTE &that) {
  if (that.numCalculations != numCalculations || that.numStates != numStates) {
    Error("The 2 VectorBTE must be aligned for dot() to work.");
  }
  if (that.dimensionality != 3 ) {
    Error("VectorBTE dot is implemented for 3D vectors only");
  }
  Eigen::MatrixXd result(statisticsSweep.getNumCalculations(),3);
  result.setZero();

  auto parallelIrrStates = bandStructure.parallelIrrStateIterator();
  size_t numParallelIrrStates = parallelIrrStates.size();

#pragma omp declare reduction (+: Eigen::MatrixXd: omp_out=omp_out+omp_in)\
     initializer(omp_priv=Eigen::MatrixXd::Zero(omp_orig.rows(),omp_orig.cols()))

#pragma omp parallel for reduction(+ : result)
  for (size_t iis=0; iis<numParallelIrrStates; iis++) {
    int is = parallelIrrStates[iis];
    if (std::find(excludeIndices.begin(), excludeIndices.end(), is) !=
        excludeIndices.end()) {
      continue;
    }
    auto isIndex = StateIndex(is);
    BteIndex iBteIdx = bandStructure.stateToBte(isIndex);
    int iBte = iBteIdx.get();
    auto rotationsStar = bandStructure.getRotationsStar(isIndex);
    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      for (const Eigen::Matrix3d &rot : rotationsStar) {
        Eigen::Vector3d x = Eigen::Vector3d::Zero();
        Eigen::Vector3d y = Eigen::Vector3d::Zero();
        for (int i : {0,1,2}) {
          for (int j : {0, 1, 2}) {
            x(i) += rot(i,j) * operator()(iCalc, j, iBte);
            y(i) += rot(i,j) * that(iCalc, j, iBte);
          }
        }
        for (int i : {0,1,2}) {
          result(iCalc,i) += x(i) * y(i);
        }
      }
    }
  }
  mpi->allReduceSum(&result);
  return result;
}

VectorBTE VectorBTE::baseOperator(VectorBTE &that, const int &operatorType) {

  VectorBTE newPopulation(statisticsSweep, bandStructure, dimensionality);

  // check that they have the same number of states
  if(this->numStates != that.numStates) {
    Error("Cannot operate on vectorBTEs with different numStates!");
  }

  // check that they have the same number of calcs (mu, T, not dim)
  if(statisticsSweep.getNumCalculations() !=
                that.statisticsSweep.getNumCalculations()) {
    Error("Cannot operate on vectorBTEs with different numCalcs!");
  }

  if (dimensionality == that.dimensionality) {

    if (operatorType == operatorSums) {
      newPopulation.data << this->data.array() + that.data.array();
    } else if (operatorType == operatorDivs) {
      newPopulation.data << this->data.array() / that.data.array();
    } else if (operatorType == operatorProd) {
      newPopulation.data << this->data.array() * that.data.array();
    } else if (operatorType == operatorDiff) {
      newPopulation.data << this->data.array() - that.data.array();
    } else {
      Error("Developer error: Operator type for VectorBTE not recognized");
    }
  // this adds the same vectorBTE that to every calc in
  // this vector BTE
  } else if (that.dimensionality == 1) {

    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      auto tup = loc2Glob(iCalc);
      auto imu = std::get<0>(tup);
      auto it = std::get<1>(tup); // temperature
      auto i2 = that.glob2Loc(imu, it, CartIndex(0)); //cartesian index

      if (operatorType == operatorSums) {
        newPopulation.data.row(iCalc) =
            this->data.row(iCalc).array() + that.data.row(i2).array();
      } else if (operatorType == operatorDivs) {
        newPopulation.data.row(iCalc) =
            this->data.row(iCalc).array() / that.data.row(i2).array();
      } else if (operatorType == operatorProd) {
        newPopulation.data.row(iCalc) =
            this->data.row(iCalc).array() * that.data.row(i2).array();
      } else if (operatorType == operatorDiff) {
        newPopulation.data.row(iCalc) =
            this->data.row(iCalc).array() - that.data.row(i2).array();
      } else {
        Error("Operator type for VectorBTE not recognized");
      }
    }
  } else {
    Error("Developer error: VectorBTE objects "
                "cannot be operated on when dim > 1.");
  }
  for (const int &iBte : excludeIndices) {
    newPopulation.data.col(iBte).setZero();
  }
  return newPopulation;
}

// product operator overload
VectorBTE VectorBTE::operator*(VectorBTE &that) {
  return baseOperator(that, operatorProd);
}

// product operator overload
VectorBTE VectorBTE::operator*(const double &scalar) {
  VectorBTE newPopulation(statisticsSweep, bandStructure, dimensionality);
  for (int i = 0; i < numCalculations; i++) {
    newPopulation.data.row(i) = this->data.row(i) * scalar;
  }
  return newPopulation;
}

// product operator overload
VectorBTE VectorBTE::operator*(const Eigen::MatrixXd &vector) {
  VectorBTE newPopulation(statisticsSweep, bandStructure, dimensionality);
  if (vector.rows() != statisticsSweep.getNumCalculations() || vector.cols() != 3) {
    Error("VectorBTE * unexpected alignment with MatrixXd");
  }
#pragma omp parallel for
  for (int iBte=0; iBte<numStates; iBte++) {
    for (int i : {0, 1, 2}) {
      for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
        newPopulation(iCalc, i, iBte) = operator()(iCalc, i, iBte) * vector(iCalc,i);
      }
    }
  }
  return newPopulation;
}

// product operator overload
VectorBTE VectorBTE::operator*(ParallelMatrix<double> &matrix) {

  if (numCalculations != dimensionality) {
    // you'd need to keep in memory a lot of matrices.
    Error("We didn't implement VectorBTE * matrix for numCalculations > 1");
  }
  if (matrix.rows() != numStates) {
    Error("VectorBTE and Matrix not aligned");
  }
  VectorBTE newPopulation(statisticsSweep, bandStructure, dimensionality);
  newPopulation.data.setZero();

  auto allLocalStates = matrix.getAllLocalStates();
  size_t numAllLocalStates = allLocalStates.size();

  #pragma omp parallel
  {
    Eigen::MatrixXd dataPrivate = newPopulation.data;
#pragma omp for
    for (size_t iTup=0; iTup<numAllLocalStates; iTup++) {
      auto tup = allLocalStates[iTup];
      auto i = std::get<0>(tup);
      auto j = std::get<1>(tup);
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
	dataPrivate(iCalc, j) += data(iCalc, i) * matrix(i, j);
      }
    }
#pragma omp critical
    for (int j=0; j<data.cols(); ++j) {
      for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
	newPopulation.data(iCalc, j) += dataPrivate(iCalc, j);
      }
    }
  }
  mpi->allReduceSum(&newPopulation.data);
  return newPopulation;
}

// sum operator overload
VectorBTE VectorBTE::operator+(VectorBTE &that) {
  return baseOperator(that, operatorSums);
}

// product operator overload
VectorBTE VectorBTE::operator-(VectorBTE &that) {
  return baseOperator(that, operatorDiff);
}

// difference operator overload
VectorBTE VectorBTE::operator-() {
  VectorBTE newPopulation(statisticsSweep, bandStructure, dimensionality);
  newPopulation.data = -this->data;
  return newPopulation;
}

// division operator overload
VectorBTE VectorBTE::operator/(VectorBTE &that) {
  return baseOperator(that, operatorDivs);
}

VectorBTE VectorBTE::sqrt() {
  VectorBTE newPopulation(statisticsSweep, bandStructure, dimensionality);
  newPopulation.data << this->data.array().sqrt();
  return newPopulation;
}

VectorBTE VectorBTE::reciprocal() {
  VectorBTE newPopulation(statisticsSweep, bandStructure, dimensionality);
  #pragma omp parallel for
  for (int iBte = 0; iBte < numStates; iBte++) {
    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      for (int iDim = 0; iDim < dimensionality; iDim++) {
        // if the linewidth is somehow zero, we should leave the recip value as
        // zero so that we don't count these states.
        if( VectorBTE::operator()(iCalc, iDim, iBte) != 0.) {
          newPopulation(iCalc, iDim, iBte) = 1./VectorBTE::operator()(iCalc, iDim, iBte);
        }
      }
    }
  }
  return newPopulation;
}

void VectorBTE::canonical2Population() {
  auto particle = bandStructure.getParticle();
#pragma omp parallel for
  for (int iBte = 0; iBte < numStates; iBte++) {
    BteIndex iBteIdx = BteIndex(iBte);
    StateIndex isIdx = bandStructure.bteToState(iBteIdx);
    double en = bandStructure.getEnergy(isIdx);
    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      auto temp = statisticsSweep.getCalcStatistics(iCalc).temperature;
      auto chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
      double pop = particle.getPopPopPm1(en, temp, chemPot);
      for (int iDim : {0,1,2}) {
        VectorBTE::operator()(iCalc, iDim, iBte) *= pop;
      }
    }
  }
}

void VectorBTE::population2Canonical() {
  auto particle = bandStructure.getParticle();
  if (particle.isFermi()) {
    Error("Possible divergence in population2Canonical");
  }
#pragma omp parallel for
  for (int iBte = 0; iBte < numStates; iBte++) {
    BteIndex iBteIdx = BteIndex(iBte);
    StateIndex isIdx = bandStructure.bteToState(iBteIdx);
    double en = bandStructure.getEnergy(isIdx);
    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      auto temp = statisticsSweep.getCalcStatistics(iCalc).temperature;
      auto chemPot = statisticsSweep.getCalcStatistics(iCalc).chemicalPotential;
      double pop = particle.getPopPopPm1(en, temp, chemPot);
      for (int iDim : {0,1,2}) {
        VectorBTE::operator()(iCalc, iDim, iBte) /= pop;
      }
    }
  }
}


// get/set operator
double &VectorBTE::operator()(const int &iCalc, const int &iDim,
                              const int &iState) {

  //if(iCalc * dimensionality + iDim >= numCalculations || iCalc * dimensionality + iDim < 0) {
  //  DeveloperError("Out of bounds calculation indexing in VectorBTE.");
  //}
  //if(iState >= numStates || iState < 0) {
  //  DeveloperError("Out of bounds state index in VectorBTE.");
  //}
  return data(iCalc * dimensionality + iDim, iState);
}

// const get/set operator
const double &VectorBTE::operator()(const int &iCalc, const int &iDim,
                                    const int &iState) const {
  return data(iCalc * dimensionality + iDim, iState);
}

int VectorBTE::glob2Loc(const ChemPotIndex &imu, const TempIndex &it,
                        const CartIndex &iDim) const {
  int i = compress3Indices(imu.get(), it.get(), iDim.get(), numChemPots,
                           numTemps, dimensionality);
  return i;
}

std::tuple<ChemPotIndex, TempIndex, CartIndex>
VectorBTE::loc2Glob(
    const int &i) const {
  auto tup = decompress3Indices(i, numChemPots, numTemps, dimensionality);
  auto imu = std::get<0>(tup);
  auto it = std::get<1>(tup);
  auto iDim = std::get<2>(tup);
  return std::make_tuple(ChemPotIndex(imu), TempIndex(it), CartIndex(iDim));
}

void VectorBTE::setConst(const double &constant) {
  data.setConstant(constant);
}

int VectorBTE::getNumStates() {return numStates;}
