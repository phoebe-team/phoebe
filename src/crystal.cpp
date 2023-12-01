#include "crystal.h"
#include "constants.h"
#include "exceptions.h"
#include "spglib.h"
#include "utilities.h"
#include "eigen.h"
#include "mpi/mpiHelper.h"
#include "periodic_table.h"
#include <iomanip>

double calcVolume(const Eigen::Matrix3d &directUnitCell) {
  Eigen::Vector3d a1 = directUnitCell.row(0);
  Eigen::Vector3d a2 = directUnitCell.row(1);
  Eigen::Vector3d a3 = directUnitCell.row(2);
  double volume;
  volume = abs(a1.dot((a2.cross(a3))));
  volume += abs(a2.dot((a3.cross(a1))));
  volume += abs(a3.dot((a1.cross(a2))));
  volume /= 3.;
  return volume;
}

Crystal::Crystal(Context &context, Eigen::Matrix3d &directUnitCell_,
                 Eigen::MatrixXd &atomicPositions_,
                 Eigen::VectorXi &atomicSpecies_,
                 std::vector<std::string> &speciesNames_,
                 Eigen::VectorXd &speciesMasses_) {

  setDirectUnitCell(directUnitCell_); // sets both direct and reciprocal
  volumeUnitCell = calcVolume(directUnitCell);

  if (volumeUnitCell <= 0.) {
    Error("Unexpected non positive volume");
  }

  dimensionality = context.getDimensionality();

  if (atomicSpecies_.size() != atomicPositions_.rows()) {
    Error("atomic species and positions are not aligned");
  }
  if (atomicPositions_.cols() != 3) {
    Error("atomic positions need three coordinates");
  }
  if ((int)speciesMasses_.size() != (int)speciesNames_.size()) {
    Error("species masses and names are not aligned");
  }

  atomicSpecies = atomicSpecies_;
  atomicPositions = atomicPositions_;
  speciesMasses = speciesMasses_;
  speciesNames = speciesNames_;

  numAtoms = int(atomicPositions.rows());
  numSpecies = int(speciesNames.size());
  speciesIsotopeCouplings.resize(numSpecies);
  Eigen::VectorXi speciesZNumber(numSpecies);

  {
    PeriodicTable pTable;
    int i = 0;
    for (std::string speciesName : speciesNames) {
      double g = pTable.getMassVariance(speciesName);
      speciesZNumber(i) = pTable.getIonicCharge(speciesName);
      speciesIsotopeCouplings(i) = g;
      i++;
    }
  }

  // We allow the user to optionally specify masses and isotopic scattering
  {
    Eigen::VectorXd customMasses = context.getMasses();
    if (customMasses.size()>0) {

      std::string s1 = context.getAppName();
      std::string s2 = "phonon";
      if (s1.find(s2) == std::string::npos) {
        Error("Can only change masses for phonon* apps");
        // I think that for el-ph transport, the mass should already be scaled
        // at the ab-initio level, or it may give rise to some inconsistencies
      }

      if (customMasses.size()!= numSpecies) {
        Error("If specifying the masses, must specify them for all"
              " atomic species");
      }
      for (int i = 0; i < numSpecies; i++) {
        if (customMasses(i) > 0.) {
          speciesMasses(i) = customMasses(i);
        }
      }
    }

    //--------------------------

    Eigen::VectorXd customCouplings = context.getIsotopeCouplings();
    for (int i=0; i<numSpecies; i++) {
      if (customCouplings.size() > 0) {
        if (customCouplings.size() != numSpecies) {
          Error("If specifying the isotopic couplings, must specify "
                "them for all atomic species");
        }
        if (customCouplings(i) > 0.) {
          speciesIsotopeCouplings(i) = customCouplings(i);
        }
      }
    }
  }

  atomicMasses.resize(numAtoms);
  atomicNames.resize(numAtoms);
  atomicIsotopeCouplings.resize(numAtoms);
  for (int i = 0; i < numAtoms; i++) {
    atomicMasses(i) = speciesMasses(atomicSpecies(i));
    atomicNames[i] = speciesNames[atomicSpecies(i)];
    atomicIsotopeCouplings(i) = speciesIsotopeCouplings(atomicSpecies(i));
  }
  // call the function to set up crystal symmetries
  generateSymmetryInformation(context);
  if (mpi->mpiHead()) {  // we only really want to print this on construction 
    std::cout << "Found " << numSymmetries << " symmetries\n";
  }
}

void Crystal::generateSymmetryInformation(Context &context) {

  int maxSize = 192;
  int rotations[maxSize][3][3];
  double translations[maxSize][3];

  if (context.getUseSymmetries()) {

    // We now look for the symmetry operations of the crystal
    // in this implementation, we rely on spglib

    // Declare and allocate c-style arrays for spglib calls
    double latticeSPG[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        // note: directUnitCell has lattice vectors along rows (a1 = cell.row(0))
        latticeSPG[i][j] = directUnitCell(i, j);
      }
    }

    // note: spglib wants fractional positions
    double(*positionSPG)[3];
    allocate(positionSPG, numAtoms);
    Eigen::Vector3d positionCrystal;
    Eigen::Vector3d positionCartesian;
    for (int i = 0; i < numAtoms; i++) {
      positionCartesian = atomicPositions.row(i);
      positionCrystal = directUnitCell.inverse() * positionCartesian;
      for (int j = 0; j < 3; j++) {
        positionSPG[i][j] = positionCrystal(j);
      }
    }

    // also wants integer types >= 1
    int *typesSPG;
    allocate(typesSPG, numAtoms);
    for (int i = 0; i < numAtoms; i++) {
      typesSPG[i] = atomicSpecies(i) + 1;
    }

    double symmetryPrecision = 1e-4;
    numSymmetries =
        spg_get_symmetry(rotations, translations, maxSize, latticeSPG,
                         positionSPG, typesSPG, numAtoms, symmetryPrecision);

    // need to explicitly deallocate allocated arrays.
    delete[] typesSPG;
    delete[] positionSPG;

    if (numSymmetries == 0) {
      Error("SPGlib failed at recognizing symmetries");
    }

  } else { // if we disable symmetries, and just use the identity

    numSymmetries = 1;
    for ( int i : {0,1,2}) {
      for ( int j : {0,1,2}) {
        rotations[0][i][j] = 0.;
      }
    }
    for ( int i : {0,1,2}) {
      translations[0][i] = 0.;
      rotations[0][i][i] = 1.;
    }
  }

  // If the system has no symmetries, we disable them
  if (numSymmetries==1) {
    context.setUseSymmetries(false);
  }

  // store the symmetries inside the class
  // note: spglib returns rotation and translation in fractional coordinates
  for (int iSymmetry = 0; iSymmetry < numSymmetries; iSymmetry++) {
    Eigen::Vector3d thisTranslation;
    thisTranslation(0) = translations[iSymmetry][0];
    thisTranslation(1) = translations[iSymmetry][1];
    thisTranslation(2) = translations[iSymmetry][2];
    Eigen::Matrix3d thisMatrix;
    for (int i : {0,1,2}) {
      for (int j : {0,1,2}) {
        thisMatrix(j, i) = rotations[iSymmetry][i][j]; // note the transpose
      }
    }
    SymmetryOperation s = {thisMatrix, thisTranslation};
    symmetryOperations.push_back(s);
  }
}

// empty constructor
Crystal::Crystal() {
  volumeUnitCell = 0.;
  numAtoms = 0;
  numSpecies = 0;
  dimensionality = 0.;
  numSymmetries = 0;
}

// copy constructor
Crystal::Crystal(const Crystal &obj) {
  directUnitCell = obj.directUnitCell;
  reciprocalUnitCell = obj.reciprocalUnitCell;
  volumeUnitCell = obj.volumeUnitCell;
  numAtoms = obj.numAtoms;
  numSpecies = obj.numSpecies;
  dimensionality = obj.dimensionality;
  atomicPositions = obj.atomicPositions;
  atomicSpecies = obj.atomicSpecies;
  atomicNames = obj.atomicNames;
  atomicMasses = obj.atomicMasses;
  atomicIsotopeCouplings = obj.atomicIsotopeCouplings;
  speciesIsotopeCouplings = obj.speciesIsotopeCouplings;
  speciesNames = obj.speciesNames;
  speciesMasses = obj.speciesMasses;
  symmetryOperations = obj.symmetryOperations;
  numSymmetries = obj.numSymmetries;
}

// assignment operator
Crystal &Crystal::operator=(const Crystal &obj) {
  if (this != &obj) {
    directUnitCell = obj.directUnitCell;
    reciprocalUnitCell = obj.reciprocalUnitCell;
    volumeUnitCell = obj.volumeUnitCell;
    numAtoms = obj.numAtoms;
    numSpecies = obj.numSpecies;
    dimensionality = obj.dimensionality;
    atomicPositions = obj.atomicPositions;
    atomicSpecies = obj.atomicSpecies;
    atomicNames = obj.atomicNames;
    atomicMasses = obj.atomicMasses;
    atomicIsotopeCouplings = obj.atomicIsotopeCouplings;
    speciesIsotopeCouplings = obj.speciesIsotopeCouplings;
    speciesNames = obj.speciesNames;
    speciesMasses = obj.speciesMasses;
    symmetryOperations = obj.symmetryOperations;
    numSymmetries = obj.numSymmetries;
  }
  return *this;
}

void Crystal::print() {
  if(!mpi->mpiHead()) return;
  // print the lattice vectors
  std::cout << "\nDirect lattice vectors (ang)" << std::endl;
  std::cout << "       a1     |     a2     |    a3" << std::endl;
  for(int i = 0; i<3; i++) {
    std::cout << std::fixed << std::setprecision(8);
    std::cout << std::setw(13) << directUnitCell(i,0)*distanceBohrToAng;
    std::cout << std::setw(13) << directUnitCell(i,1)*distanceBohrToAng;
    std::cout << std::setw(13) << directUnitCell(i,2)*distanceBohrToAng;
    std::cout << "\n";
  }
  // print the atomic positions
  std::cout << "Atomic Positions (Cartesian, ang)" << std::endl;
  for(int i = 0; i<numAtoms; i++) {
    std::cout << std::setw(3) << speciesNames[atomicSpecies[i]];
    std::cout << std::setw(13) << atomicPositions(i,0)*distanceBohrToAng;
    std::cout << std::setw(13) << atomicPositions(i,1)*distanceBohrToAng;
    std::cout << std::setw(13) << atomicPositions(i,2)*distanceBohrToAng;
    std::cout << "\n";
  }
  std::cout << std::endl;
}

Eigen::Matrix3d
Crystal::calcReciprocalCell(const Eigen::Matrix3d &directUnitCell) {
  Eigen::Matrix3d reciprocalCell = twoPi * directUnitCell.inverse().transpose();
  return reciprocalCell;
}

void Crystal::setDirectUnitCell(const Eigen::Matrix3d &directUnitCell_) {
  directUnitCell = directUnitCell_;
  reciprocalUnitCell = calcReciprocalCell(directUnitCell);
}

const Eigen::Matrix3d &Crystal::getDirectUnitCell() { return directUnitCell; }

const Eigen::Matrix3d &Crystal::getReciprocalUnitCell() {
  // note: reciprocalUnitCell is  in units of twoPi
  // i.e. must be multiplied by twoPi
  return reciprocalUnitCell;
}

const int &Crystal::getNumAtoms() const { return numAtoms; }

double Crystal::getVolumeUnitCell(int dimensionality_) {
  double volume;
  if (dimensionality_ == 3) {
    volume = volumeUnitCell;
  } else if (dimensionality_ == 2) {
    volume = abs(directUnitCell(0, 0) * directUnitCell(1, 1) -
                 directUnitCell(0, 1) * directUnitCell(1, 0));
  } else {
    volume = directUnitCell(2, 2);
  }
  return volume;
}

const Eigen::MatrixXd &Crystal::getAtomicPositions() { return atomicPositions; }

const Eigen::VectorXi &Crystal::getAtomicSpecies() { return atomicSpecies; }

const std::vector<std::string> &Crystal::getAtomicNames() {
  return atomicNames;
}

const Eigen::VectorXd &Crystal::getAtomicMasses() { return atomicMasses; }

const Eigen::VectorXd &Crystal::getAtomicIsotopeCouplings() {
  return atomicIsotopeCouplings;
}

const std::vector<std::string> &Crystal::getSpeciesNames() {
  return speciesNames;
}

const Eigen::VectorXd &Crystal::getSpeciesMasses() { return speciesMasses; }

const std::vector<SymmetryOperation> &Crystal::getSymmetryOperations() {
  return symmetryOperations;
}

const int &Crystal::getNumSymmetries() const { return numSymmetries; }

int Crystal::getDimensionality() const { return dimensionality; }

int Crystal::getNumSpecies() const { return numSpecies; }

std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
Crystal::buildWignerSeitzVectors(const Eigen::Vector3i &grid,
                                 const int &superCellFactor) {

  int nx = superCellFactor;

  std::vector<Eigen::Vector3d> tmpVectors;
  std::vector<int> tmpDegeneracies;

  // the point at midIndex is the reference "n" vector
  // i.e. the one generated by i0=i1=i2=0
  // if it's the minimum vector, than it's in the
  // Wigner Seitz zone of super cell A.
  // Therefore we save this vector.
  int midIndex = pow(2*(nx + 1)+1,3);
  midIndex /= 2;

  // what are we doing:
  // the "n" specifies a grid of lattice vectors, that are (2*(cutoff+1)+1)^3
  // times bigger than the grid of wavevectors. a grid of lattice vectors
  // equal to the grid of wavevectors would be the bare minimum for the
  // interpolation to work, and wouldn't be enough for anything good.

  // now, we loop over the vectors of super cell A, build a vector r.
  for (int i0 = -nx * grid(0); i0 <= nx * grid(0); i0++) {
    for (int i1 = -nx * grid(1); i1 <= nx * grid(1); i1++) {
      for (int i2 = -nx * grid(2); i2 <= nx * grid(2); i2++) {
        // loop over a super cell B of size ((searchSize+1)*2+1)^3
        // bigger than super cell A. We compute the distance between any
        // vector in super cell B w.r.t. the vector "n"

        // We calculate |r-R|^2
        std::vector<double> distances;
        for (int j0 = -nx - 1; j0 <= nx + 1; j0++) {
          for (int j1 = -nx - 1; j1 <= nx + 1; j1++) {
            for (int j2 = -nx - 1; j2 <= nx + 1; j2++) {
              Eigen::Vector3d dist;
              dist(0) = i0 - j0 * grid(0);
              dist(1) = i1 - j1 * grid(1);
              dist(2) = i2 - j2 * grid(2);
              // distances in cartesian space
              dist = directUnitCell * dist;
              double dist2 = dist.norm();
              distances.push_back(dist2);
            }
          }
        }

        // find the minimum distance out of all distances
        double distMin = *min_element(distances.begin(), distances.end());

        if (abs(distances[midIndex] - distMin) < 1.0e-6) {
          // count its degeneracy
          int degeneracy = 0;
          for (double dist : distances) {
            if (abs(dist - distMin) < 1.0e-6) { ++degeneracy; }
          }
          tmpDegeneracies.push_back(degeneracy);

          // store vector
          Eigen::Vector3d thisVec = {double(i0), double(i1), double(i2)};
          tmpVectors.push_back(thisVec);
        }
      }
    }
  }

  // check that we found all vectors
  double tot = 0.;
  for (int x : tmpDegeneracies) {
    tot += 1. / double(x);
  }
  if (abs(tot - grid(0) * grid(1) * grid(2)) > 1.0e-6) {
    std::cout << grid(0) * grid(1) * grid(2) << " " << tot << "\n";
    Error("Completeness check failed in buildWignerSeitzVectors");
  }

  // now we store the list of these lattice vectors in the class members
  auto numPositionVectors = int(tmpVectors.size());
  Eigen::VectorXd positionDegeneracies(numPositionVectors);
  Eigen::MatrixXd positionVectors(3, numPositionVectors);
  int originIndex = -1; // to look for R=0 vector
  for (int iR = 0; iR < numPositionVectors; iR++) {
    auto thisVec = tmpVectors[iR];
    // we convert from crystal to cartesian coordinates
    positionVectors.col(iR) = directUnitCell * thisVec;
    positionDegeneracies(iR) = tmpDegeneracies[iR];
    //
    if (thisVec.norm() < 1.0e-6) {
      originIndex = iR;
    }
  }
  if (originIndex==-1) Error("R=0 index not found");

  // As a convention, we keep the origin vectors at the index iR = 0
  // so we swap it. The other vectors aren't in any particular order.
  double tmp1 = positionDegeneracies(originIndex);
  double tmp2 = positionDegeneracies(0);
  positionDegeneracies(0) = tmp1;
  positionDegeneracies(originIndex) = tmp2;
  Eigen::Vector3d tmpV1 = positionVectors.col(originIndex);
  Eigen::Vector3d tmpV2 = positionVectors.col(0);
  positionVectors.col(0) = tmpV1;
  positionVectors.col(originIndex) = tmpV2;

  return std::make_tuple(positionVectors, positionDegeneracies);
}

std::tuple<Eigen::MatrixXd, Eigen::Tensor<double, 3>>
Crystal::buildWignerSeitzVectorsWithShift(const Eigen::Vector3i &grid,
                                          const Eigen::MatrixXd &shift,
                                          const int &superCellFactor) {

  // Note: we expect shift to be in cartesian coordinates!

  if (shift.rows() != 3) {
    Error("shift should have exactly 3 cartesian coordinates");
  }
  auto shiftDims = int(shift.cols());

  int nx = superCellFactor;

  int maxVectors = 2 * nx * grid(0) + 1;
  maxVectors *= 2 * nx * grid(1) + 1;
  maxVectors *= 2 * nx * grid(2) + 1;
  Eigen::MatrixXd tmpNumPoints(shiftDims, shiftDims);
  Eigen::Tensor<double, 3> tmpDegeneraciesAll(shiftDims, shiftDims, maxVectors);
  Eigen::Tensor<double, 4> tmpVectorsAll(shiftDims, shiftDims, maxVectors, 3);

  // what are we doing:
  // the "n" specifies a grid of lattice vectors, that are (2*(cutoff+1)+1)^3
  // times bigger than the grid of wavevectors. a grid of lattice vectors
  // equal to the grid of wavevectors would be the bare minimum for the
  // interpolation to work, and wouldn't be enough for anything good.

  for (int iDim = 0; iDim < shiftDims; iDim++) {
    for (int jDim = 0; jDim < shiftDims; jDim++) {
      // algorithm as before, except the shift

      std::vector<Eigen::Vector3d> tmpVectors;
      std::vector<double> tmpDegeneracies;

      // now, we loop over the vectors of super cell A, build a vector r.
      for (int i0 = -nx * grid(0); i0 <= nx * grid(0); i0++) {
        for (int i1 = -nx * grid(1); i1 <= nx * grid(1); i1++) {
          for (int i2 = -nx * grid(2); i2 <= nx * grid(2); i2++) {
            // loop over a super cell B of size ((searchSize+1)*2+1)^3
            // bigger than super cell A. We compute the distance between any
            // vector in super cell B w.r.t. the vector "n"

            // We calculate |r-R|^2
            std::vector<double> distances;
            for (int j0 = -nx - 1; j0 <= nx + 1; j0++) {
              for (int j1 = -nx - 1; j1 <= nx + 1; j1++) {
                for (int j2 = -nx - 1; j2 <= nx + 1; j2++) {
                  Eigen::Vector3d dist;
                  dist(0) = i0 - j0 * grid(0);
                  dist(1) = i1 - j1 * grid(1);
                  dist(2) = i2 - j2 * grid(2);
                  // distances in cartesian space
                  dist = directUnitCell * dist;

                  dist += shift.col(jDim) - shift.col(iDim);

                  double dist2 = dist.norm();
                  distances.push_back(dist2);
                }
              }
            }

            // find the minimum distance out of all distances
            double distMin = distances[0]; // [0] is just the first guess
            for (double dist : distances) {
              if (dist < distMin) {
                distMin = dist;
              }
            }

            // the point at midIndex is the reference "n" vector
            // i.e. the one generated by i0=i1=i2=0
            // if it's the minimum vector, than it's in the
            // Wigner Seitz zone of super cell A.
            // Therefore we save this vector.
            unsigned int midIndex = distances.size() / 2;
            if (abs(distances[midIndex] - distMin) < 1.0e-6) {
              // count its degeneracy
              double degeneracy = 0.;
              for (double dist : distances) {
                if (abs(dist - distMin) < 1.0e-6) {
                  degeneracy += 1.;
                }
              }

              tmpDegeneracies.push_back(degeneracy);

              // store vector
              Eigen::Vector3d thisVec;
              thisVec << i0, i1, i2;
              tmpVectors.push_back(thisVec);
            }
          }
        }
      }

      tmpNumPoints(iDim, jDim) = int(tmpVectors.size());
      for (unsigned int iR = 0; iR < tmpVectors.size(); iR++) {
        tmpDegeneraciesAll(iDim, jDim, iR) = tmpDegeneracies[iR];
        for (int i : {0, 1, 2}) {
          tmpVectorsAll(iDim, jDim, iR, i) = tmpVectors[iR](i);
        }
      }
    }
  }

  // find the unique set of vectors, so it's a smaller list
  std::vector<Eigen::Vector3d> tmpVectors;
  for (int iDim = 0; iDim < shiftDims; iDim++) {
    for (int jDim = 0; jDim < shiftDims; jDim++) {
      for (int iR = 0; iR < tmpNumPoints(iDim, jDim); iR++) {
        Eigen::Vector3d x;
        for (int i : {0, 1, 2}) {
          x(i) = tmpVectorsAll(iDim, jDim, iR, i);
        }
        bool found = false;
        if (tmpVectors.empty())
          found = true; // if empty, add vector
        for (const Eigen::Vector3d &y : tmpVectors) {
          if ((x - y).squaredNorm() < 1.0e-6) {
            found = true;
            break;
          }
        }
        if (found) {
          tmpVectors.push_back(x);
        }
      }
    }
  }
  int numVectors = int(tmpVectors.size());

  // save the degeneracies in a smaller tensor
  Eigen::Tensor<double, 3> degeneracies(shiftDims, shiftDims, numVectors);
  degeneracies.setZero(); // important!
  for (int iDim = 0; iDim < shiftDims; iDim++) {
    for (int jDim = 0; jDim < shiftDims; jDim++) {
      for (int iR = 0; iR < tmpNumPoints(iDim, jDim); iR++) {
        Eigen::Vector3d x;
        for (int i : {0, 1, 2}) {
          x(i) = tmpVectorsAll(iDim, jDim, iR, i);
        }
        double deg = tmpDegeneraciesAll(iDim, jDim, iR);
        int iRNew = 0;
        for (const Eigen::Vector3d &y : tmpVectors) {
          if ((x - y).squaredNorm() < 1.0e-6) {
            degeneracies(iDim, jDim, iRNew) = deg;
            break;
          }
          ++iRNew;
        }
      }
    }
  }

  // convert to cartesian coordinates
  Eigen::MatrixXd bravaisVectors(3, numVectors);
  for (int iR = 0; iR < numVectors; iR++) {
    auto thisVec = tmpVectors[iR];
    // we convert from crystal to cartesian coordinates
    bravaisVectors.col(iR) = directUnitCell * thisVec;
  }

  // check that we found all vectors
  for (int iDim = 0; iDim < shiftDims; iDim++) {
    for (int jDim = 0; jDim < shiftDims; jDim++) {
      double tot = 0.;
      for (int iR = 0; iR < numVectors; iR++) {
        if (degeneracies(iDim, jDim, iR) > 0) {
          tot += 1. / degeneracies(iDim, jDim, iR);
        }
      }
      if (abs(tot - grid(0) * grid(1) * grid(2)) > 1.0e-6) {
        Error("Failed completeness check in buildWignerSeitzVectorsWithShift");
      }
    }
  }
  return std::make_tuple(bravaisVectors, degeneracies);
}
