#ifndef BAND_STRUCTURE_H
#define BAND_STRUCTURE_H

#include "exceptions.h"
#include "particle.h"
#include "points.h"
#include "utilities.h"
#include "Matrix.h"
#include <utility>

/** Base class for describing objects containing the band structure, i.e.
 * the harmonic properties of a quasiparticle, as a function of wavevectors
 * (Points).
 */
class BaseBandStructure {
 public:
 
  /** Base destructor for bandstructure class, silences warnings */
  //virtual ~BaseBandStructure() = default;
 
  /** Get the Particle object associated with this class
   * @return particle: a Particle object, describing e.g. whether this
   * is a phonon or electron bandStructure
   */
  virtual Particle getParticle() = 0;

  /** Returns the wavevectors on which the band structure is computed.
   * @return Points: the object representing the Brillouin zone wavevectors.
   */
  virtual Points getPoints() = 0;

  /** Returns a wavevector, given a wavevector index.
   * The wavevector index runs from 0 to numPoints-1
   */
  virtual Point getPoint(const int &pointIndex) = 0;

  /** Returns the total number of k/q-points.
   * @param useFullGrid: default = false. If true, returns the number of
   * points of the full monkhorst-pack grid of wavevectors, otherwise,
   * returns the number of wavevectors stored in the BandStructure.
   * @return numPoints: the total number of wavevectors of the bandStructure.
   */
  virtual int getNumPoints(const bool &useFullGrid = false) = 0;

  /** Returns the number of bands.
   * @return numBands: the total number of bands present in the bandstructure.
   * If the number of bands is not constant, calls an error.
   */
  virtual int getNumBands() = 0;

  /** Returns the number of bands. If the band structure is active,
   * and there are a different number of bands at each wavevector, this function
   * returns the number of bands at that point. If not, this defaults
   * to the behavior of getNumBands() and just returns the number of bands.
   * This is necessary for parts of the code where we leave an option to
   * swap in different child band structure types.
   */
  virtual int getNumBands(WavevectorIndex &ik) = 0;

  /** Checks whether the bandStructure has been built discarding some Bloch
   * states from those available.
   * @return windowMethod: one of the values of Window::filterMethod. 0 for
   * no filter.
   */
  virtual int hasWindow() = 0;
  /** Returns the boolean determining if this band structure is distributed
   * or not.
   */
  virtual bool getIsDistributed() = 0;

  // needed in the BTE
  /** Builds a Bloch state index, which combines both wavevector index and
   * band index.
   * It's used to view the various matrices such as energy as a 1D vector,
   * and can be used in combination with get() methods.
   * @param wavevectorIndex: strong-typed index on wavevector
   * @return stateIndex: integer from 0 to numStates-1=numBands*numPoints-1
   */
  virtual int getIndex(const WavevectorIndex &ik, const BandIndex &ib) = 0;

  /** Given a Bloch state index, finds the corresponding wavevector and band
   * index.
   * @param stateIndex: StateIndex integer from 0 to numStates - 1 =
   * numBands*numPoints-1
   * @return WavevectorIndex: strong-typed index on wavevector
   * @return BandIndex: strong-typed index on bands
   */
  virtual std::tuple<WavevectorIndex, BandIndex> getIndex(const int &is) = 0;
  virtual std::tuple<WavevectorIndex, BandIndex> getIndex(StateIndex &is) = 0;

  /** Returns the total number of Bloch states.
   * @return numStates: the integer number of Bloch states.
   */
  virtual int getNumStates() = 0;

  /** Returns an iterator to be used for loops over the Bloch state index.
   * The values of the iterator are distributed in N blocks over N MPI ranks.
   */
  std::vector<size_t> parallelStateIterator();

  /** Returns the energy of a quasiparticle from its Bloch index
   * Used for accessing the band structure in the BTE.
   * @param stateIndex: an integer index in range [0,numStates[
   * @return energy: the value of the QP energy for that given Bloch index.
   * Phonon energies are referred to zero, with negative energies being
   * actually complex phonon frequencies. Electronic energies are not saved
   * with any particular reference, and should be used together with the
   * chemical potential computed by StatisticsSweep. By policy, it's in
   * rydberg units.
   */
  virtual const double &getEnergy(StateIndex &is) = 0;
  virtual Eigen::VectorXd getEnergies(WavevectorIndex &ik) = 0;

  /** Return the maximum energy of a bandstructure.
  * Used when ph energy maximum is a cutoff for phel scattering.
  * @return maxEnergy: maximum energy value of bandstructure in Ry
  */
  virtual double getMaxEnergy() = 0;

  /** Returns the energy of a quasiparticle from its Bloch index
   * Used for accessing the band structure in the BTE.
   * @param stateIndex: an integer index in range [0,numStates[
   * @return velocity: a 3d vector with velocity. By policy, we save it in
   * the cartesian basis and in atomic rydberg units.
   */
  virtual Eigen::Vector3d getGroupVelocity(StateIndex &is) = 0;
  virtual Eigen::MatrixXd getGroupVelocities(WavevectorIndex &ik) = 0;
  virtual Eigen::Tensor<std::complex<double>, 3> getVelocities(
      WavevectorIndex &ik) = 0;

  virtual Eigen::MatrixXcd getEigenvectors(WavevectorIndex &ik) = 0;
  virtual Eigen::Tensor<std::complex<double>, 3> getPhEigenvectors(
      WavevectorIndex &ik) = 0;

  /** Returns the energy of a quasiparticle from its Bloch index
   * Used for accessing the band structure in the BTE.
   * @param stateIndex: an integer index in range [0,numStates[
   * @return wavevector: a 3d vector with the wavevector in cartesian
   * coordinates in units of Bohr^-1.
   */
  virtual Eigen::Vector3d getWavevector(StateIndex &is) = 0;
  virtual Eigen::Vector3d getWavevector(WavevectorIndex &ik) = 0;

  /** Method to save quasiparticle eigenvectors inside FullBandStructure().
   * @param point: a vector of 3 crystal coordinates. The method will look
   * for the wavevector index.
   * @param energies: a vector of size (numBands) with the quasiparticle
   * energies
   */
  virtual void setEnergies(Point &point, Eigen::VectorXd &energies_) = 0;

  /** Method to save quasiparticle eigenvectors inside FullBandStructure().
   * Note that in this case, eigenvectors are passed as a matrix, which is
   * the case e.g. for the Wannier interpolation, where the eigenvectors
   * represent the unitary transformation matrix U for Wannier localization.
   * @param point: a Point object with the coordinates of the wavevector,
   * which should come from the same Point class stored in FullBandStructure
   * @param eigenvectors: a complex matrix of size (numBands,numBands)
   */
  virtual void setEigenvectors(Point &point,
                               Eigen::MatrixXcd &eigenvectors_) = 0;

  /** Saves in the class the velocities computed at a particular point.
   * @param point: a Point object representing the wavevector where these
   * velocities have been computed.
   * @param velocities: a rank-3 tensor of size (numBands,numBands,3)
   * containing the matrix elements of the velocity operator. Diagonal
   * elements are the quasiparticle group velocities.
   */
  virtual void setVelocities(
      Point &point, Eigen::Tensor<std::complex<double>, 3> &velocities_) = 0;

  /** Given a irreducible point index, find the list of rotations to reconstruct
   * the equivalent points.
   *
   * @param ikIndex: Index of the irreducible wavevector. This index is a
   * number between 0 and N_k_reducible.
   * @return rotations: a vector with the rotations used to reconstruct the
   * symmetry-equivalent Bloch states.
   */
  virtual std::vector<Eigen::Matrix3d> getRotationsStar(WavevectorIndex &ikIndex) = 0;

  /** Given an irreducible Bloch state (i.e. any band at an irreducible point),
   * find the list of rotations to reconstruct the equivalent points.
   *
   * @param isIndex: Index of the irreducible Bloch State. This index is a
   * number between 0 and numStates.
   * @return rotations: a vector with the rotations used to reconstruct the
   * symmetry-equivalent Bloch states.
   */
  virtual std::vector<Eigen::Matrix3d> getRotationsStar(StateIndex &isIndex) = 0;

  /** Given a point in crystal or cartesian coordinates, returns the index of
   * the irreducible point and the rotation such that
   * rotation*irrPoint = redPoint
   *
   * @param x: point coordinates
   * @param basis: either Points::crystalCoordinates or Points::cartesianCoordinates,
   * this will treat x in the appropriate coordinate. Also the returned rotation
   * will be in the corresponding basis.
   * @return <ik,rot>: a tuple with the index of the irreducible point and the
   * rotation matrix connecting the irreducible and reducible point.
   */
  virtual std::tuple<int, Eigen::Matrix3d> getRotationToIrreducible(
      const Eigen::Vector3d &x,
      const int &basis = Points::crystalCoordinates) = 0;

  /** Utility method to convert an index over Bloch states in the band structure
   * into a Bloch state index usable by VectorBTE.
   * If a state is not mapped to the VectorBTE, throws an error.
   *
   * @param StateIndex: the index of the Bloch state in the band structure.
   * @return BteIndex: index of the Bloch state in the BTE
   */
  virtual BteIndex stateToBte(StateIndex &isIndex) = 0;

  /** Utility method to convert an index over Bloch states in a VectorBTE into
   * the Bloch state index in the band structure.
   * Unlike stateToBte, this should always have a solution.
   *
   * @param iBteIndex: index of the Bloch state in the BTE
   * @return StateIndex: the index of the Bloch state in the band structure.
   */
  virtual StateIndex bteToState(BteIndex &iBteIndex) = 0;

  /** Iterator over the Bloch states in the band structure, over just the
   * irreducible wavevectors, but isn't distributed over MPI processes.
   *
   * @return State-indices: a vector<int> with the indices over Bloch states
   * stored in the band structure
   */
  virtual std::vector<int> irrStateIterator() = 0;

  /** Iterator over the Bloch states in the band structure, distributed over
   * MPI processes, running only over irreducible wavevectors.
   *
   * @return State-indices: a vector<int> with the indices over Bloch states
   * stored in the band structure
   */
  virtual std::vector<int> parallelIrrStateIterator() = 0;

  /** Iterator over the irreducible points indices.
   * The iterator is serial, not parallelized with MPI.
   *
   * @return k-indices: a std::vector<int> with the indices of the irreducible
   * points.
   */
  virtual std::vector<int> irrPointsIterator() = 0;

  /** Iterator over the irreducible points indices.
   * The iterator is parallelized over MPI processes.
   *
   * @return k-indices: a std::vector<int> with the indices of the irreducible
   * points.
   */
  virtual std::vector<int> parallelIrrPointsIterator() = 0;

  /** Find the index of a point in the reducible list of points, given its
   * coordinates in the crystal basis.
   *
   * @param crystalCoordinates: coordinates of the point in crystal basis
   * @param suppressError: default false. If false, will throw an error if the
   * point is not found
   * @return ik: the index of the point
   */
  virtual int getPointIndex(const Eigen::Vector3d &crystalCoordinates,
                             const bool &suppressError=false) = 0;

  /** Method to find the points equivalent to an irreducible point.
   *
   * @param ik: index of the irreducible point, with ik running on the full list
   * of reducible points.
   * @return vector<int>: the list of indices of the reducible points
   * equivalent to point #ik.
   */
  virtual std::vector<int> getReducibleStarFromIrreducible(const int &ik) = 0;
};

class ActiveBandStructure;
// forward declaration of friend class
// we could remove this by making ActiveBS a subclass of FullBS

/** FullBandStructure is the class that stores the energies, velocities and
 * eigenvectors of a quasiparticle computed on a set of wavevectors (as defined
 * by Points() ) in the Brillouin zone.
 * By default, each MPI process holds a full copy of the band structure.
 * However, the band structure can be distributed over the wavevectors, if so
 * specified in the constructor.
 *
 * An important note for developers: When using a distributed band structure,
 * looping over numStates of the band structure will not work -- you need to
 * loop over the iterator of indices provided by getStateIndices or
 * getWavevectorIndices. The class will throw errors when nonlocal values are
 * being requested. All functions in band structure are written to take the
 * global wavevector indices associated with the Points object internal to the
 * band structure (because we use the Point class to find wavevector indices in
 * get and set functions of band structure).
 */
class FullBandStructure : public BaseBandStructure {
 public:
  /** Constructor of the FullBandStructure
   * @param numBands: an integer with the number of bands in the system
   * @param particle: a Particle object that contains the type of
   * quasiparticle
   * @param withVelocities: a boolean to decide whether to store velocities
   * @param withEigenvectors: a boolean to decide whether to store eigenVectors
   * @param points: the underlying mesh of wavevectors.
   * @param isDistributed: if true, we distribute in memory the
   * storage of quantities, parallelizing over wavevectors (points). By default
   * we don't distribute data in parallel.
   */
  FullBandStructure(int numBands_, Particle &particle_, bool withVelocities,
                    bool withEigenvectors, Points &points_,
                    bool isDistributed_ = false);

  //FullBandStructure();

  /** Copy constructor
   */
  FullBandStructure(const FullBandStructure &that);

  /** Assignment operator
   */
  FullBandStructure &operator=(const FullBandStructure &that);

  /** Get the Particle object associated with this class
   * @return particle: a Particle object, describing e.g. whether this
   * is a phonon or electron bandStructure
   */
  Particle getParticle() override;

  /** Returns the wavevectors on which the band structure is computed.
   * @return Points: the object representing the Brillouin zone wavevectors.
   */
  Points getPoints() override;

  /** Returns a wavevector, given a wavevector index.
   * The wavevector index runs from 0 to numPoints-1
   */
  Point getPoint(const int &pointIndex) override;

  /** Returns the total number of k/q-points.
   * @param useFullGrid: used for compatibility with the ActiveBandStructure,
   * it doesn't affect the returned value.
   * @return numPoints: the total number of wavevectors of the bandStructure.
   */
  int getNumPoints(const bool &useFullGrid = false) override;

  /** Returns the number of bands.
   * @return numBands: the total number of bands in the bandStructure.
   */
  int getNumBands() override;
  /** Returns the number of bands, to provide flexibility in cases where
   * full or activeBandStructure could be used.
   * @return numBands: the total number of bands in the bandStructure.
   */
  int getNumBands(WavevectorIndex &ik) override;

  int hasWindow() override;

  bool getIsDistributed() override;

  bool getHasEigenvectors();

  /** Builds a Bloch state index, which runs on both wavevector index and
   * band index. ik runs from 0 to numPoints-1, ib from 0 to numBands-1.
   * It's used to view the various matrices such as energy as a 1D vector,
   * and can be used in combination with get() methods.
   * @param wavevectorIndex: strong-typed index on wavevector
   * @return stateIndex: integer from 0 to numStates-1=numBands*numPoints-1
   */
  int getIndex(const WavevectorIndex &ik, const BandIndex &ib) override;

  /** Given a Bloch state index, finds the corresponding wavevector and band
   * index.
   * @param stateIndex: integer from 0 to numStates-1=numBands*numPoints-1
   * @return WavevectorIndex: strong-typed index on wavevector
   * @return BandIndex: strong-typed index on bands
   */
  std::tuple<WavevectorIndex, BandIndex> getIndex(const int &is) override;

  /** Given a Bloch state index, finds the corresponding wavevector and band
   * index.
   * @param stateIndex: StateIndex(is) object where is is an integer from 0 to
   * numStates-1=numBands*numPoints-1
   * @return WavevectorIndex: strong-typed index on wavevector
   * @return BandIndex: strong-typed index on bands
   */
  std::tuple<WavevectorIndex, BandIndex> getIndex(StateIndex &is) override;

  /** Returns the total number of Bloch states, equal to numPoints*numBands.
   * @return numStates: the total number of Bloch states in the class.
   */
  int getNumStates() override;

  /** Returns the indices of all wavevector indices on this process, or in
  * an undistributed case, returns all wavevector indices.
  * @return wavevectorIndices: a vector of wavevector indices for use as
  * an iterator.
  */
  std::vector<int> getWavevectorIndices();

  /** Returns the indices of all state indices on this process, or in
  * an undistributed case, returns all state indices.
  * @return stateIndices: a vector of tuples of (ib,ik) indices for use as
  * an iterator.
  */
  std::vector<std::tuple<WavevectorIndex, BandIndex>> getStateIndices();

  /** Returns the indices of all bands on this process, or in an
  * undistributed case, returns all band indices.
  * @return bandIndices: a vector of band indices for use as an iterator.
  */
  std::vector<int> getBandIndices() const;

  /** Returns the energy of a quasiparticle from its Bloch index.
   * Same as getEnergy(const int &stateIndex), but using a StateIndex input
   * @param stateIndex: a StateIndex(is) object where 'is' is an integer
   * running over the number of states [0,numStates-1].
   * @return energy: the value of the QP energy for that given Bloch index.
   * Phonon energies are referred to zero, with negative energies being
   * actually complex phonon frequencies. Electronic energies are not saved
   * with any particular reference, and should be used together with the
   * chemical potential computed by StatisticsSweep. By policy, it's in
   * rydberg units.
   */
  const double &getEnergy(StateIndex &is) override;

  /** Returns the energy of a quasiparticle from its band and wavevector index.
   * Same as getEnergy(const int &stateIndex), but using ib,ik instead.
   * ik should always be the global wavevector index, or this will be wrong!
   * @param ik: the wavevector index of the particle state
   * @param ib: the band index of the particle state
   * @return energy: the value of the QP energy for that given Bloch index.
   * Phonon energies are referred to zero, with negative energies being
   * actually complex phonon frequencies. Electronic energies are not saved
   * with any particular reference, and should be used together with the
   * chemical potential computed by StatisticsSweep. By policy, it's in
   * rydberg units.
   */
  const double &getEnergy(WavevectorIndex &ik, BandIndex &ib);

  /** Returns the energies of all quasiparticle computed at a specified
   * wavevector.
   * @param wavevectorIndex: a WavevectorIndex(ik) where ik is an integer
   * index running over the wavevectors [0,numPoints-1]
   * @return energies: an Eigen vector of quasiparticle energies for that
   * given wavevector.
   * Phonon energies are referred to zero, with negative energies being
   * actually complex phonon frequencies. Electronic energies are not saved
   * with any particular reference, and should be used together with the
   * chemical potential computed by StatisticsSweep. In rydberg units.
   */
  Eigen::VectorXd getEnergies(WavevectorIndex &ik) override;

  /** Return the maximum energy of a bandstructure.
  * Used when ph energy maximum is a cutoff for phel scattering.
  * @return maxEnergy: maximum energy value of bandstructure in Ry
  */
  double getMaxEnergy() override;

  /** Returns the group velocity of a quasiparticle from its Bloch index.
   * Used for accessing the band structure in the BTE.
   * @param stateIndex: a StateIndex(is) object where 'is' is an integer index
   * in the range [0,numStates[
   * @return velocity: a 3d vector with velocity. By policy, we save it in
   * the cartesian basis and in atomic rydberg units.
   */
  Eigen::Vector3d getGroupVelocity(StateIndex &is) override;

  /** Returns the group velocity of a quasiparticle for all bands at a
   * specified wavevector index.
   * Used for accessing the band structure in the BTE.
   * @param wavevectorIndex: a WavevectorIndex(ik) object where 'ik' is an
   * integer index running over the wavevectors in range [0,numPoints-1]
   * @return velocity: a matrix(numBands,3) with the group velocity, in
   * cartesian basis and in atomic rydberg units.
   */
  Eigen::MatrixXd getGroupVelocities(WavevectorIndex &ik) override;

  /** Returns the velocity operator (including off-diagonal matrix elements)
   * of the quasiparticles at the specified wavevector index.
   * Used for accessing the band structure in the BTE.
   * @param wavevectorIndex: a WavevectorIndex(ik) object where 'ik' is an
   * integer index running over the wavevectors in range [0,numPoints-1]
   * @return velocity: a tensor (numBands,numBands,3) with the velocity
   * operator matrix elements, in cartesian basis and in atomic rydberg units.
   */
  Eigen::Tensor<std::complex<double>, 3> getVelocities(WavevectorIndex &ik) override;

  /** Obtain the eigenvectors of the quasiparticles at a specified wavevector.
   * @param wavevectorIndex: a WavevectorIndex(ik) object where ik is the
   * integer wavevector index running over [0,numPoints-1].
   * @return eigenvectors: a complex matrix(numBands,numBands) where numBands
   * is the number of Bloch states present at the specified wavevector.
   * Eigenvectors are ordered along columns.
   * Note that all band structure interpolations may give eigenvectors.
   */
  Eigen::MatrixXcd getEigenvectors(WavevectorIndex &ik) override;

  /** Obtain the eigenvectors of the quasiparticles at a specified wavevector.
   * It's only meaningful for the phonon band structure, where eigenvectors
   * are more naturally represented in this shape!
   * @param wavevectorIndex: a WavevectorIndex(ik) object where ik is the
   * integer wavevector index running over [0,numPoints-1].
   * @return eigenvectors: a complex tensor (3,numAtoms,numBands) where
   * numBands is the number of Bloch states present at the specified
   * wavevector, numAtoms is the number of atoms in the crystal unit cell and
   * 3 is a cartesian directions.
   */
  Eigen::Tensor<std::complex<double>, 3> getPhEigenvectors(WavevectorIndex &ik) override;

  /** Returns the energy of a quasiparticle from its Bloch index.
   * @param stateIndex: a StateIndex(is) object where 'is' is an integer
   * index in range [0,numStates-1].
   * @return wavevector: a 3d vector with the wavevector in cartesian
   * coordinates in units of Bohr^-1.
   */
  Eigen::Vector3d getWavevector(StateIndex &is) override;

  /** Returns the energy of a quasiparticle from its Bloch index.
   * @param wavevectorIndex: a WavevectorIndex(ik) object where 'ik' is an
   * integer index in range [0,numPoints-1].
   * @return wavevector: a 3d vector with the wavevector in cartesian
   * coordinates in units of Bohr^-1.
   */
  Eigen::Vector3d getWavevector(WavevectorIndex &ik) override;

  /** Method to save quasiparticle energies inside FullBandStructure().
   * @param point: a point object, which also provides the wavevector index.
   * @param energies: a vector of size (numBands) with the quasiparticle
   * energies
   */
  void setEnergies(Point &point, Eigen::VectorXd &energies_) override;

  /** Method to save quasiparticle eigenvectors inside FullBandStructure().
   * Note that in this case, eigenvectors are passed as a matrix, which is
   * the case e.g. for the Wannier interpolation, where the eigenvectors
   * represent the unitary transformation matrix U for Wannier localization.
   * @param point: a Point object with the coordinates of the wavevector,
   * which should come from the same Point class stored in FullBandStructure
   * @param eigenvectors: a complex matrix of size (numBands,numBands)
   */
  void setEigenvectors(Point &point, Eigen::MatrixXcd &eigenvectors_) override;

  /** Saves in the class the velocities computed at a particular point.
   * @param point: a Point object representing the wavevector where these
   * velocities have been computed.
   * @param velocities: a rank-3 tensor of size (numBands,numBands,3)
   * containing the matrix elements of the velocity operator. Diagonal
   * elements are the quasiparticle group velocities.
   */
  void setVelocities(Point &point,
                     Eigen::Tensor<std::complex<double>, 3> &velocities_) override;

  /** Method to save quasiparticle eigenvectors inside FullBandStructure().
   * @param point: a vector of 3 crystal coordinates. The method will look
   * for the wavevector index.
   * @param energies: a vector of size (numBands) with the quasiparticle
   * energies
   */
  void setEnergies(Eigen::Vector3d &point, Eigen::VectorXd &energies_);

  /** Returns all electronic energies for all wavevectors at fixed band index
   * Used by the Fourier interpolation of the band structure.
   * @param bandIndex: index in [0,numBands[ for the quasiparticle band
   * @return energies: a vector of size numPoints with the QP energies.
   * Energies are ordered as the underlying wavevectors in Points.
   */
  Eigen::VectorXd getBandEnergies(int &bandIndex);

  /** Given a irreducible point index, find the list of rotations to reconstruct
   * the equivalent points.
   *
   * @param ikIndex: Index of the irreducible wavevector. This index is a
   * number between 0 and N_k_reducible.
   * @return rotations: a vector with the rotations used to reconstruct the
   * symmetry-equivalent Bloch states.
   */
  std::vector<Eigen::Matrix3d> getRotationsStar(WavevectorIndex &ikIndex) override;

  /** Given an irreducible Bloch state (i.e. any band at an irreducible point),
   * find the list of rotations to reconstruct the equivalent points.
   *
   * @param isIndex: Index of the irreducible Bloch State. This index is a
   * number between 0 and numStates.
   * @return rotations: a vector with the rotations used to reconstruct the
   * symmetry-equivalent Bloch states.
   */
  std::vector<Eigen::Matrix3d> getRotationsStar(StateIndex &isIndex) override;

  /** Given a point in crystal or cartesian coordinates, returns the index of
   * the irreducible point and the rotation such that
   * rotation*irrPoint = redPoint
   *
   * @param x: point coordinates
   * @param basis: either Points::crystalCoordinates or Points::cartesianCoordinates,
   * this will treat x in the appropriate coordinate. Also the returned rotation
   * will be in the corresponding basis.
   * @return <ik,rot>: a tuple with the index of the irreducible point and the
   * rotation matrix connecting the irreducible and reducible point.
   */
  std::tuple<int, Eigen::Matrix3d> getRotationToIrreducible(
      const Eigen::Vector3d &x,
      const int &basis = Points::crystalCoordinates) override;

  /** Utility method to convert an index over Bloch states in the band structure
   * into a Bloch state index usable by VectorBTE.
   * If a state is not mapped to the VectorBTE, throws an error.
   *
   * @param StateIndex: the index of the Bloch state in the band structure.
   * @return BteIndex: index of the Bloch state in the BTE
   */
  BteIndex stateToBte(StateIndex &isIndex) override;

  /** Utility method to convert an index over Bloch states in a VectorBTE into
   * the Bloch state index in the band structure.
   * Unlike stateToBte, this should always have a solution.
   *
   * @param iBteIndex: index of the Bloch state in the BTE
   * @return StateIndex: the index of the Bloch state in the band structure.
   */
  StateIndex bteToState(BteIndex &iBteIndex) override;

  /** Iterator over the Bloch states in the band structure, over just the
   * irreducible wavevectors, but isn't distributed over MPI processes.
   *
   * @return State-indices: a vector<int> with the indices over Bloch states
   * stored in the band structure
   */
  std::vector<int> irrStateIterator() override;

  /** Iterator over the Bloch states in the band structure, distributed over
   * MPI processes, running only over irreducible wavevectors.
   *
   * @return State-indices: a vector<int> with the indices over Bloch states
   * stored in the band structure
   */
  std::vector<int> parallelIrrStateIterator() override;

  /** Iterator over the irreducible points indices.
   * The iterator is serial, not parallelized with MPI.
   *
   * @return k-indices: a std::vector<int> with the indices of the irreducible
   * points.
   */
  std::vector<int> irrPointsIterator() override;

  /** Iterator over the irreducible points indices.
   * The iterator is parallelized over MPI processes.
   *
   * @return k-indices: a std::vector<int> with the indices of the irreducible
   * points.
   */
  std::vector<int> parallelIrrPointsIterator() override;

  /** Find the index of a point in the reducible list of points, given its
   * coordinates in the crystal basis.
   *
   * @param crystalCoordinates: coordinates of the point in crystal basis
   * @param suppressError: default false. If false, will throw an error if the
   * point is not found
   * @return ik: the index of the point
   */
  int getPointIndex(const Eigen::Vector3d &crystalCoordinates,
                    const bool &suppressError=false) override;

  /** Method to find the points equivalent to an irreducible point.
   *
   * @param ik: index of the irreducible point, with ik running on the full list
   * of reducible points.
   * @return vector<int>: the list of indices of the reducible points
   * equivalent to point #ik.
   */
  std::vector<int> getReducibleStarFromIrreducible(const int &ik) override;
protected:
  // stores the quasiparticle kind
  Particle particle;

  // link to the underlying mesh of points
  Points &points;  // these may be FullPoints or PathPoints

  // if the bands are distributed or not
  bool isDistributed = false;

  bool hasEigenvectors = false;
  bool hasVelocities = false;

  // matrices storing the raw data
  Matrix<double> energies;                    // size(bands,points)
  Matrix<std::complex<double>> velocities;    // size(3*bands^2,points)
  Matrix<std::complex<double>> eigenvectors;  // size(bands^2,points)

  // auxiliary variables
  int numBands = 0;
  int numAtoms = 0;
  int numPoints = 0;
  int numLocalPoints = 0;

  // method to find the index of the point, from its crystal coordinates
  int getIndex(Eigen::Vector3d &pointCoordinates);

  // ActiveBandStructure, which restricts the FullBandStructure to a subset
  // of wavevectors, needs low-level access to the raw data (for now)
  friend class ActiveBandStructure;
};

#endif
