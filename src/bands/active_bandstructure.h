#ifndef ACT_BAND_STRUCTURE_H
#define ACT_BAND_STRUCTURE_H

#include "bandstructure.h"
#include "constants.h"
#include "harmonic.h"
#include "particle.h"
#include "points.h"
#include "statistics_sweep.h"
#include "window.h"

/** Class container of the quasiparticle band structure, i.e. energies,
 * velocities, eigenvectors and wavevectors.
 * In contrast to FullBandStructure, which stores this information for all
 * wavevectors and bands, ActiveBandStructure stores this data for a specified
 * subset of wavevectors, and a subset of bands which may change from one
 * wavevector to another.
 * Should be initialized through the static method provided below, which makes
 * use of the Window that is selected by user input.
 * The subset of wavevectors is specified using an ActivePoints class.
 */
class ActiveBandStructure : public BaseBandStructure {
public:
  /** Almost empty constructor, to be used internally.
   */
  ActiveBandStructure(Particle &particle_, Points &points_);

  /** Copy constructor
   */
  ActiveBandStructure(const ActiveBandStructure &that);

  /** Assignment operator
   */
  ActiveBandStructure &operator=(const ActiveBandStructure &that);

  /** Default constructor.
   * Populate the band structure for the subset of wavevectors specified by
   * points, but saving all bands for such wavevector.
   * Using this constructor, we populate the band structure on the wavevectors
   * specified by points.
   * Currently used for the caching of the 3rd wavevectors info in the
   * 3-phonon scattering matrix.
   */
  ActiveBandStructure(const Points &points_,
                      HarmonicHamiltonian *h0, const bool &withEigenvectors,
                      const bool &withVelocities);

  /** Get the Particle object associated with this class
   * @return particle: a Particle object, describing e.g. whether this
   * is a phonon or electron bandStructure
   */
  Particle getParticle() override;

  /** Returns the wavevectors on which the band structure is computed.
   * @return Points: the object representing the Brillouin zone wavevectors.
   * Should be an ActivePoints class.
   */
  Points getPoints() override;

  /** Returns a wavevector, given a wavevector index.
   * The wavevector index runs from 0 to numPoints-1, where numPoints is the
   * number of active wavevectors to be used in the calculations.
   */
  Point getPoint(const int &pointIndex) override;

  /** Returns the total number of k/q-points.
   * @param useFullGrid: default = false. If true, returns the number of
   * points of the full monkhorst-pack grid of wavevectors, otherwise,
   * returns the number of wavevectors stored in the BandStructure.
   * In particular for this class, the number of wavevectors stored in the list
   * is obtained with false, if true, and if activePoints.parentPoints is a
   * FullPoints object, then we will return the number of points in the
   * original full grid of points (useful when normalizing equations).
   * @return numPoints: the total number of wavevectors of the bandStructure.
   */
  int getNumPoints(const bool &useFullGrid = false) override;

  /** Returns the number of bands.
   * If the number of bands is not constant, it raises an error.
   * In that case, it's best to check the size of the objects returned by
   * the get*() methods or use getNumBands(ik).
   * @return numBandsFull: the total number of bands of the bandStructure
   *    (only returned in the case where the bands have not been filtered.)
   */
  int getNumBands() override;

  /** Returns the number of bands at a given wavevector.
   * @return numBands: the number of bands at the requested ik.
   */
  int getNumBands(WavevectorIndex &ik) override;

  /** Checks whether the bandStructure has been built discarding some Bloch
   * states from those available.
   * @return windowMethod: one of the values of Window::filterMethod. 0 for
   * no filter.
   */
  int hasWindow() override;

  /** Returns if this band structure is distributed. In the case of
   * activeBandStructure, currently always returns false.
   */
  bool getIsDistributed() override;

  /** Builds a Bloch state index, which runs on both wavevector index and
   * band index.
   * @param wavevectorIndex: strong-typed index on wavevector index
   * @param bandIndex: strong-typed index on band index. Note that bandIndex
   * has different range for every wavevector
   * @return stateIndex: integer from 0 to numStates-1
   */
  int getIndex(const WavevectorIndex &ik, const BandIndex &ib) override;

  /** Given a Bloch state index, finds the corresponding wavevector and band
   * index.
   * @param stateIndex: integer from 0 to numStates-1
   * @return WavevectorIndex: a strong-typed index on wavevector
   * @return BandIndex: a strong-typed index on bands
   */
  std::tuple<WavevectorIndex, BandIndex> getIndex(const int &is) override;

  /** Given a Bloch state index, finds the corresponding wavevector and band
   * index.
   * @param stateIndex: a StateIndex(is) object where is is an integer from 0
   * to numStates-1
   * @return WavevectorIndex: strong-typed index on wavevector
   * @return BandIndex: strong-typed index on bands
   */
  std::tuple<WavevectorIndex, BandIndex> getIndex(StateIndex &is) override;

  /** Returns the total number of Bloch states, equal to numPoints*numBands.
   * @return numStates: the total number of Bloch states in the class.
   */
  int getNumStates() override;

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

  /** Returns the maximum energy value. This can be useful when
   * setting energy scales based on which phonons are discarded
   * (as in the phel scattering, where the window is set relative
   * to the maximum phonon energies)
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
   * @return velocity: a matrix(numActiveBands,3) with the group velocity, in
   * cartesian basis and in atomic rydberg units, where numActiveBands is the
   * number of active bands present at the specified wavevector.
   */
  Eigen::MatrixXd getGroupVelocities(WavevectorIndex &ik) override;

  /** Returns the velocity operator (including off-diagonal matrix elements)
   * of the quasiparticles at the specified wavevector index.
   * Used for accessing the band structure in the BTE.
   * @param wavevectorIndex: a WavevectorIndex(ik) object where 'ik' is an
   * integer index running over the wavevectors in range [0,numPoints-1]
   * @return velocity: a tensor (numActiveBands,numActiveBands,3) with the
   * velocity operator matrix elements, in cartesian basis and in atomic rydberg
   * units, where numActiveBands is the number of active bands present at the
   * specified wavevector.
   */
  Eigen::Tensor<std::complex<double>, 3> getVelocities(WavevectorIndex &ik) override;

  /** Obtain the eigenvectors of the quasiparticles at a specified wavevector.
   * @param wavevectorIndex: a WavevectorIndex(ik) object where ik is the
   * integer wavevector index running over [0,numPoints-1].
   * @return eigenvectors: a complex matrix(numBands,numActiveBands) where
   * numBands is the full (before applying the window) number of Bloch states
   * present at the specified wavevector and numActiveBands is the number of
   * filtered bands at the desired wavevector.
   * Eigenvectors are ordered along columns.
   * Note that all band structure interpolations may give eigenvectors.
   */
  Eigen::MatrixXcd getEigenvectors(WavevectorIndex &ik) override;

  /** Obtain the eigenvectors of the quasiparticles at a specified wavevector.
   * It's only meaningful for the phonon band structure, where eigenvectors
   * are more naturally represented in this shape!
   * @param wavevectorIndex: a WavevectorIndex(ik) object where ik is the
   * integer wavevector index running over [0,numPoints-1].
   * @return eigenvectors: a complex tensor (3,numAtoms,numActiveBands) where
   * numActiveBands is the number of Bloch states present at the specified
   * wavevector, numAtoms is the number of atoms in the crystal unit cell and
   * 3 are the cartesian directions.
   */
  Eigen::Tensor<std::complex<double>, 3> getPhEigenvectors(WavevectorIndex &ik) override;

  /** Returns the energy of a quasiparticle from its Bloch index
   * Used for accessing the band structure in the BTE.
   * @param stateIndex: an integer index in range [0,numStates[
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

  /** Method to save quasiparticle energies inside ActiveBandStructure().
   * @param point: a point object, which will be used to lookup the wavevector
   * index ik.
   * @param energies: a vector of size (numBands(ik)) with the quasiparticle
   * energies for that point.
   */
  void setEnergies(Point &point, Eigen::VectorXd &energies_) override;

  /** Method to save quasiparticle energies inside ActiveBandStructure().
   * @param point: a point object, which will be used to lookup the wavevector
   * index ik.
   * @param energies: a vector of size (numBands(ik)) with the quasiparticle
   * energies for that point.
   */
  void setEnergies(Point &point, std::vector<double> &energies_);

  /** Method to save quasiparticle eigenvectors inside ActiveBandStructure.
   * @param point: a point object, which will be used to lookup the wavevector
   * index ik.
   * @param eigenvectors: a matrix of size (numFullBands, numBands(ik)) with
   * the quasiparticle eigenvectors for this point.
   */
  void setEigenvectors(Point &point, Eigen::MatrixXcd &eigenvectors_) override;

  /** Method to save quasiparticle velocity operator in ActiveBandStructure.
   * @param point: a point object, which will be used to lookup the wavevector
   * index ik.
   * @param velocities: a tensor of size (numBands(ik),numBands(ik),3) with
   * the quasiparticle velocity matrix elements for this wavevector.
   */
  void setVelocities(Point &point,
                     Eigen::Tensor<std::complex<double>, 3> &velocities_) override;

  /** Preferred method to initialize the ActiveBandStructure class.
   * @param context: object with user input variables.
   * @param h0: hamiltonian object to be diagonalized.
   * @param points: initial unfiltered list of wavevectors, which will be
   * filtered into an ActivePoints object.
   * @param withEigenvectors: compute and store the eigenvectors
   * @param withVelocities: compute and store the velocity matrix elements
   * @param forceBuildAPP: forces activeBandStructure to be built
   * using the internal buildAsPostprocessing method, even if the input
   * H0 is for phonons.
   */
  static std::tuple<ActiveBandStructure, StatisticsSweep>
  builder(Context &context, HarmonicHamiltonian &h0, Points &points,
          const bool &withEigenvectors = true,
          const bool &withVelocities = true, const bool &forceBuildAPP = false);

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
      const Eigen::Vector3d &x, const int &basis = Points::crystalCoordinates) override;

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
                     const bool &suppressError = false) override;

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

  Points points;

  // note: we don't store a matrix: we are storing an object (Nk,Nb),
  // with a variable number of bands Nb per point
  std::vector<double> energies;
  std::vector<std::complex<double>> velocities;
  std::vector<std::complex<double>> eigenvectors;

  bool hasEigenvectors = false;
  int numStates = 0;
  int numIrrStates;
  int numIrrPoints;
  int numPoints;

  Eigen::VectorXi numBands;
  int numFullBands = 0;
  int windowMethod = 0;

  // index management
  // these are two auxiliary vectors to store indices
  Eigen::MatrixXi auxBloch2Comb;
  Eigen::VectorXi cumulativeKbOffset;
  Eigen::MatrixXi bteAuxBloch2Comb;
  Eigen::VectorXi bteCumulativeKbOffset;
  Eigen::VectorXi cumulativeKbbOffset;
  // this is the functionality to build the indices
  void buildIndices(); // to be called after building the band structure
  // and these are the tools to convert indices
  void buildSymmetries();

  // utilities to convert Bloch indices into internal indices
  int velBloch2Comb(const int &ik, const int &ib1, const int &ib2,
                     const int &i);
  int eigBloch2Comb(const int &ik, const int &ibFull, const int &ibRed);
  int bloch2Comb(const int &k, const int &b);
  std::tuple<int, int> comb2Bloch(const int &is);

  int bteBloch2Comb(const int &k, const int &b);
  std::tuple<int, int> bteComb2Bloch(const int &is);

  void buildOnTheFly(Window &window, Points points_, HarmonicHamiltonian &h0,
                     const bool &withEigenvectors = true,
                     const bool &withVelocities = true);

  StatisticsSweep buildAsPostprocessing(Context &context, Points points_,
                                        HarmonicHamiltonian &h0,
                                        const bool &withEigenvector = true,
                                        const bool &withVelocities = true);
};

#endif
