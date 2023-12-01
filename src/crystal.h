#ifndef CRYSTAL_H
#define CRYSTAL_H

#include "eigen.h"
#include <string>
#include <vector>
#include "context.h"

struct SymmetryOperation {
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
};

/** Object to store the information on the crystal unit cell,
 *such as atomic positions, crystal lattice vectors, etc...
 * Note that fractional occupancies are not supported.
 */
class Crystal {
protected:
  /** utility function to invert the direct unit cell
   */
  static Eigen::Matrix3d calcReciprocalCell(const Eigen::Matrix3d &directUnitCell);

  /** Internal utility to set the crystal unit cell and the reciprocal cell
   */
  void setDirectUnitCell(const Eigen::Matrix3d &directUnitCell_);

  /** These are the internal quantities used to store
   * - lattice vectors
   * - reciprocal lattice vectors
   * - (real space) crystal unit cell volume
   * - number of atoms in unit cell
   * - number of atomic species in the crystal
   * - dimensionality (to log whether we work in 1, 2, or 3D.
   */
  Eigen::Matrix3d directUnitCell;
  Eigen::Matrix3d reciprocalUnitCell;
  double volumeUnitCell;
  int numAtoms;
  int numSpecies;
  int dimensionality;

  // these vectors/matrices  running over the number of atoms
  Eigen::MatrixXd atomicPositions;         // size (numAtoms,3)
  Eigen::VectorXi atomicSpecies;           // size (numAtoms)
  std::vector<std::string> atomicNames;    // size (numAtoms)
  Eigen::VectorXd atomicMasses;            // size (numAtoms)
  Eigen::VectorXd atomicIsotopeCouplings;  // size (numAtoms)

  // vectors running over the number of species
  std::vector<std::string> speciesNames;   // size (numSpecies)
  Eigen::VectorXd speciesMasses;           // size (numSpecies)
  Eigen::VectorXd speciesIsotopeCouplings; // size (numSpecies)

  // Untested for now
  std::vector<SymmetryOperation> symmetryOperations;
  int numSymmetries;

public:
  /** Class containing the information on the crystal structure
   * Note: a number of important quantities used throughout the code are
   * referenced here, so check if it does go out of scope.
   * @param directUnitCell: a matrix containing the crystal lattice vectors
   * in Bohr.
   * @param atomicPositions(numAtoms,3): list of atomic positions in cartesian
   * coordinates.
   * @param atomicSpecies(numAtoms): list of integers identifying the species
   * of each atom.
   * @param speciesNames(numSpecies): list of names of species. The position
   * should match the indices of the parameter atomicSpecies.
   * @param speciesMasses(numSpecies): array with the masses of each species,
   *  in rydberg.
   */
  Crystal(Context &context, Eigen::Matrix3d &directUnitCell_, Eigen::MatrixXd &atomicPositions_,
          Eigen::VectorXi &atomicSpecies_,
          std::vector<std::string> &speciesNames_,
          Eigen::VectorXd &speciesMasses_);

  /** Empty constructor.
   */
  Crystal();

  /** Copy constructor
   */
  Crystal(const Crystal &obj);

  /** Copy assignment operator
   */
  Crystal &operator=(const Crystal &obj);

  /** Print the crystal information */
  void print();

  //  Setter and getter for all the variables above

  /** Returns the crystal unit cell in real space, in Bohr.
   * Lattice vectors are rows (e.g. a0 = cell.row(0), ...)
   */
  const Eigen::Matrix3d &getDirectUnitCell();

  /** Returns the reciprocal lattice vectors, in units of Bohr^-1/tpi.
   * Lattice vectors are rows (e.g. a0 = cell.row(0), ...)
   * must be multiplied by twoPi to be complete
   */
  const Eigen::Matrix3d &getReciprocalUnitCell();

  /** get the number of atoms in the unit cell
   *
   */
  const int &getNumAtoms() const;

  /** get the volume of the crystal unit cell in Bohr^3
   * @param dimensionality: returns the volume of the unit cell on a reduced
   * dimensionality. If 2D, it is ASSUMED that the z-direction is the non
   * periodic direction. If 1D, it's assumed that z is the periodic direction
   */
  double getVolumeUnitCell(int dimensionality_ = 3);

  /** get the symmetry operations of the crystal, in cartesian coordinates.
   * Returns a vector of SymmetryOperation. A SymmetryOperation is a
   * structure containing the rotation matrix and the translation vector
   * for the given symmetry. The size of the vector is equal to
   * getNumSymmetries().
   */
  const std::vector<SymmetryOperation> &getSymmetryOperations();

  /** get the number of symmetries operations that are used by phoebe.
   * For the time being, we only retain symmetry operations that don't use a
   * translation.
   */
  const int &getNumSymmetries() const;

  /** get the atomic positions, in cartesian coordinates
   * we return an array of size (numAtoms,3)
   */
  const Eigen::MatrixXd &getAtomicPositions();

  /** get the species of each atom in the unit cell.
   * The atomic species are simply identified by an integer id
   * The vector returned has size (numAtoms)
   */
  const Eigen::VectorXi &getAtomicSpecies();

  /** get the vector of atomic names
   * vector has size (numAtoms)
   */
  const std::vector<std::string> &getAtomicNames();

  /** get the vector of atomic masses
   * the vector has size (numAtoms)
   */
  const Eigen::VectorXd &getAtomicMasses();

  /** get the vector of names of atomic species
   * the vector has size (numSpecies)
   */
  const std::vector<std::string> &getSpeciesNames();

  /** get the vector of masses of atomic species, in rydberg
   * the vector has size (numSpecies)
   */
  const Eigen::VectorXd &getSpeciesMasses();

  /** return the dimensionality of the crystal, i.e. a number from 1 to 3
   */
  int getDimensionality() const;

  /** Return the number of atomic species present in the crystal
   */
  int getNumSpecies() const;

  const Eigen::VectorXd &getAtomicIsotopeCouplings();

  /** Build the list of Bravais lattice vectors (real space) that live within
   * the Wigner Seitz zone of a super cell
   * which is grid(0) x grid(1) x grid(2) bigger than the unitCell.
   *
   * @param grid: size of the super cell for the WS construction.
   * @param superCellFactor: in order to do the correct folding of wavevectors,
   * we look for wavevectors in a slightly bigger super cell. A factor 2 should
   * be enough, but could be increased if the code fails to find all vectors.
   * @return: a tuple with bravaisLatticeVectors(3,numVectors) in cartesian
   * coordinates and their degeneracies(numVectors).
   * Note: the weights to be used for integrations are 1/degeneracies.
   */
  std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
  buildWignerSeitzVectors(const Eigen::Vector3i &grid,
                          const int &superCellFactor = 2);

  /** Similar to buildWignerSeitzVectors, we build the list of Bravais lattice
   * vectors (real space) that live within the Wigner Seitz zone of a super cell
   * which is grid(0) x grid(1) x grid(2) bigger than the unitCell.
   * However, the vectors are slightly shifted. For example, for phonons, we
   * want the vectors R+tau(3,na)-tau(3,nb), where tau are atomic positions.
   * A similar transform may be used for the Wannier transformation, where the
   * WS is centered on the Wannier-functions' centers.
   *
   * Note: for a well-converged calculation, the difference between with and
   * without shift shouldn't make a difference when used in Fourier transforms.
   * This is because the interactions should decay quickly.
   *
   * @param grid: size of the super cell for the WS construction.
   * @param shift: a shift in cartesian coordinates of shape(3,nDim).
   * @param superCellFactor: in order to do the correct folding of wavevectors,
   * we look for wavevectors in a slightly bigger super cell. A factor 2 should
   * be enough, but could be increased if the code fails to find all vectors.
   * @return: a tuple with bravaisLatticeVectors(3,numVectors) in cartesian
   * coordinates and their degeneracies(numVectors,nDim,nDim), where the last
   * two indices must be used in conjunction with the meaning of shift. Some
   * weights might be set to zero if they don't fall in the WS zone.
   * Note: the weights to be used for integrations are 1/degeneracies.
   */
  std::tuple<Eigen::MatrixXd, Eigen::Tensor<double, 3>>
  buildWignerSeitzVectorsWithShift(const Eigen::Vector3i &grid,
                                   const Eigen::MatrixXd &shift,
                                   const int &superCellFactor = 2);

  /** A function to set up the symmetry rotation matrices for a crystal, 
   * which are then stored in this object. */
  void generateSymmetryInformation(Context &context);


};

#endif
