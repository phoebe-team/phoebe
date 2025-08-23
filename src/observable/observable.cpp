#include "observable.h"
#include <cmath>

Observable::Observable(Context &context_, StatisticsSweep &statisticsSweep_,
                       Crystal &crystal_)
    : context(context_), statisticsSweep(statisticsSweep_), crystal(crystal_) {
  numCalculations = statisticsSweep.getNumCalculations();
  numChemPots = statisticsSweep.getNumChemicalPotentials();
  numTemps = statisticsSweep.getNumTemperatures();
  dimensionality = crystal.getDimensionality();
}

// copy constructor
Observable::Observable(const Observable &that)
    : context(that.context), statisticsSweep(that.statisticsSweep),
      crystal(that.crystal) {
  dimensionality = that.dimensionality;
  numChemPots = that.numChemPots;
  numTemps = that.numTemps;
  numCalculations = that.numCalculations;
  scalar = that.scalar;
  vectord = that.vectord;
  tensordxd = that.tensordxd;
  tensordxdxdxd = that.tensordxdxdxd;
}

// copy assignment
Observable &Observable::operator=(const Observable &that) {
  if (this != &that) {
    context = that.context;
    statisticsSweep = that.statisticsSweep;
    crystal = that.crystal;
    numChemPots = that.numChemPots;
    numTemps = that.numTemps;
    numCalculations = that.numCalculations;
    scalar = that.scalar;
    vectord = that.vectord;
    tensordxd = that.tensordxd;
    tensordxdxdxd = that.tensordxdxdxd;
  }
  return *this;
}

int Observable::glob2Loc(const ChemPotIndex &imu, const TempIndex &it) const {
  return compress2Indices(imu.get(), it.get(), numChemPots, numTemps);
}

std::tuple<ChemPotIndex, TempIndex> Observable::loc2Glob(const int &i) const {
  auto tup = decompress2Indices(i, numChemPots, numTemps);
  auto imu = std::get<0>(tup);
  auto it = std::get<1>(tup);
  return std::make_tuple(ChemPotIndex(imu), TempIndex(it));
}

Observable Observable::operator-(const Observable &that) {
  Observable newObservable(context, statisticsSweep, crystal);
  baseOperatorMinus(newObservable, that);
  return newObservable;
}

void Observable::baseOperatorMinus(Observable &newObservable,
                                   const Observable &that) {
  if (whichType() == isScalar) {
    for (int is = 0; is < numCalculations; is++) {
      newObservable.scalar(is) = scalar(is) - that.scalar(is);
    }
  } else if (whichType() == isVector) {
    for (int is = 0; is < numCalculations; is++) {
      for (int i = 0; i < dimensionality; i++) {
        newObservable.vectord(is, i) = vectord(is, i) - that.vectord(is, i);
      }
    }
  } else if (whichType() == is2Tensor) {
    for (int is = 0; is < numCalculations; is++) {
      for (int i = 0; i < dimensionality; i++) {
        for (int j = 0; j < dimensionality; j++) {
          newObservable.tensordxd(is, i, j) =
              tensordxd(is, i, j) - that.tensordxd(is, i, j);
        }
      }
    }
  } else if (whichType() == is4Tensor) {
    for (int is = 0; is < numCalculations; is++) {
      for (int i = 0; i < dimensionality; i++) {
        for (int j = 0; j < dimensionality; j++) {
          for (int k = 0; k < dimensionality; k++) {
            for (int l = 0; l < dimensionality; l++) {
              newObservable.tensordxdxdxd(is, i, j, k, l) =
                  tensordxdxdxd(is, i, j, k, l) -
                  that.tensordxdxdxd(is, i, j, k, l);
            }
          }
        }
      }
    }
  }
}

int Observable::whichType() { return isScalar; }

Eigen::VectorXd Observable::getNorm() {
  Eigen::VectorXd norm(numCalculations);
  norm.setZero();
  if (whichType() == isScalar) {
    for (int is = 0; is < numCalculations; is++) {
      norm(is) = abs(scalar(is));
    }
  } else if (whichType() == isVector) {
    for (int is = 0; is < numCalculations; is++) {
      for (int i = 0; i < dimensionality; i++) {
        norm(is) += vectord(is, i) * vectord(is, i);
      }
      norm(is) = sqrt(norm(is)) / double(dimensionality);
    }
  } else if (whichType() == is2Tensor) {
    for (int is = 0; is < numCalculations; is++) {
      for (int i = 0; i < dimensionality; i++) {
        for (int j = 0; j < dimensionality; j++) {
          norm(is) += tensordxd(is, i, j) * tensordxd(is, i, j);
        }
      }
      norm(is) = sqrt(norm(is)) / double(dimensionality * dimensionality);
    }
  } else if (whichType() == is4Tensor) {
    for (int is = 0; is < numCalculations; is++) {
      for (int i = 0; i < dimensionality; i++) {
        for (int j = 0; j < dimensionality; j++) {
          for (int k = 0; k < dimensionality; k++) {
            for (int l = 0; l < dimensionality; l++) {
              norm(is) +=
                  tensordxdxdxd(is, i, j, k, l) * tensordxdxdxd(is, i, j, k, l);
            }
          }
        }
      }
      norm(is) = sqrt(norm(is)) / double(pow(dimensionality, 4));
    }
  }
  return norm;
}

void Observable::symmetrize(Eigen::Tensor<double, 3>& allTransportCoeffs) { 

  // get symmetry rotations of the crystal in cartesian coords
  // in case there's no symmetries, we need to trick Phoebe into
  // generating a crystal which uses them.
  bool useSyms = context.getUseSymmetries();
  context.setUseSymmetries(true);
  Crystal symCrystal(crystal);
  symCrystal.generateSymmetryInformation(context);
  auto symOps = symCrystal.getSymmetryOperations();
  context.setUseSymmetries(useSyms);

  auto invLVs = crystal.getDirectUnitCell().inverse();
  auto LVs = crystal.getDirectUnitCell();

  for (int iCalc = 0; iCalc < numCalculations; iCalc++) {

    Eigen::Matrix3d transportCoeffs; 

    // copy the 3x3 matrix of a single calculation 
    for (int j : {0, 1, 2}) { 
      for (int i : {0, 1, 2}) {
        transportCoeffs(i,j) = allTransportCoeffs(iCalc,i,j);
      }
    }
    // to hold the symmetrized coeffs
    Eigen::Matrix3d symCoeffs;
    symCoeffs.setZero();
      
    for(SymmetryOperation symOp: symOps) {
      Eigen::Matrix3d rotation = symOp.rotation;
      rotation = LVs * rotation * invLVs; //convert to Cartesian
      Eigen::Matrix3d rotationTranspose = rotation.transpose();
      symCoeffs += rotationTranspose * transportCoeffs * rotation;
    }
    transportCoeffs = symCoeffs * (1. / symOps.size());

    // place them back into the full tensor
    for (int j : {0, 1, 2}) {
      for (int i : {0, 1, 2}) {
        allTransportCoeffs(iCalc,i,j) = transportCoeffs(i,j);
      }
    }
  }
}
