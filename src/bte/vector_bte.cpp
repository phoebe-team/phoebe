#include "vector_bte.h"
#include "constants.h"

// default constructor
VectorBTE::VectorBTE(StatisticsSweep &statisticsSweep_,
                     BaseBandStructure &bandStructure_,
                     const int &dimensionality_)
    : BaseVectorBTE(statisticsSweep_,
                    bandStructure_.irrStateIterator().size(), dimensionality_),
      bandStructure(bandStructure_) {

  if (bandStructure.getParticle().isPhonon()) {
    for (int is : bandStructure.irrStateIterator()) {
      auto isIdx = StateIndex(is);
      double en = bandStructure.getEnergy(isIdx);
      if (en < 0.1 / ryToCmm1) { // cutoff at 0.1 cm^-1
        int iBte = bandStructure.stateToBte(isIdx).get();
        excludeIndices.push_back(iBte);
      }
    }
  }
}

// copy constructor
VectorBTE::VectorBTE(const VectorBTE &that)
    : BaseVectorBTE(that), bandStructure(that.bandStructure) {}

// copy assignment
VectorBTE &VectorBTE::operator=(const VectorBTE &that) {
  BaseVectorBTE::operator=(that);
  if (this != &that) {
    bandStructure = that.bandStructure;
  }
  return *this;
}

// product operator overload
Eigen::MatrixXd VectorBTE::dot(const VectorBTE &that) {
  if (that.numCalculations != numCalculations || that.numStates != numStates) {
    Error e("The 2 VectorBTE must be aligned for dot() to work.");
  }
  if (that.dimensionality != 3 ) {
    Error("VectorBTE dot is implemented for 3D vectors only");
  }
  Eigen::MatrixXd result(statisticsSweep.getNumCalculations(),3);
  result.setZero();
  for (int is : bandStructure.parallelIrrStateIterator()) {

    if (std::find(excludeIndices.begin(), excludeIndices.end(), is) !=
        excludeIndices.end()) {
      continue;
    }

    auto isIndex = StateIndex(is);
    BteIndex iBteIdx = bandStructure.stateToBte(isIndex);
    auto rotationsStar = bandStructure.getRotationsStar(isIndex);
    for (int iCalc = 0; iCalc < statisticsSweep.getNumCalculations(); iCalc++) {
      for (Eigen::Matrix3d rot : rotationsStar) {
        Eigen::Vector3d x = Eigen::Vector3d::Zero();
        Eigen::Vector3d y = Eigen::Vector3d::Zero();
        for (int i : {0,1,2}) {
          for (int j : {0, 1, 2}) {
            x(i) += rot(i,j) * operator()(iCalc, j, iBteIdx.get());
            y(i) += rot(i,j) * that(iCalc, j, iBteIdx.get());
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
      Error e("Operator type for VectorBTE not recognized");
    }

  } else if (that.dimensionality == 1) {

    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      auto tup = loc2Glob(iCalc);
      auto imu = std::get<0>(tup);
      auto it = std::get<1>(tup);
      auto i2 = that.glob2Loc(imu, it, CartIndex(0));

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
        Error e("Operator type for VectorBTE not recognized");
      }
    }
  } else {
    Error e("VectorBTE can't handle dimensionality for this case");
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
    Error e("VectorBTE * unexpected alignment with MatrixXd");
  }
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
    Error e("We didn't implement VectorBTE * matrix for numCalculations > 1");
  }
  if (matrix.rows() != numStates) {
    Error e("VectorBTE and Matrix not aligned");
  }
  VectorBTE newPopulation(statisticsSweep, bandStructure, dimensionality);
  newPopulation.data.setZero();
  for (auto tup : matrix.getAllLocalStates()) {
    auto i = std::get<0>(tup);
    auto j = std::get<1>(tup);
    for (int iCalc = 0; iCalc < numCalculations; iCalc++) {
      newPopulation.data(iCalc, j) += data(iCalc, i) * matrix(i, j); // -5e-12
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
  newPopulation.data << 1. / this->data.array();
  return newPopulation;
}

void VectorBTE::canonical2Population() {
  auto particle = bandStructure.getParticle();
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
    Error e("Possible divergence in population2Canonical");
  }
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
