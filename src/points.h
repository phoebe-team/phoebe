#ifndef POINTS_H
#define POINTS_H

#include "crystal.h"
#include "eigen.h"
#include "exceptions.h"

const int crystalCoordinates_ = 0;
const int cartesianCoordinates_ = 1;

//// Forward declarator of Points, because Point ants to store Points as member
//// and Points has methods returning Point. Here we avoid recursive dependency.
class Points;

/** Class used to pass a single wavevector
 */
class Point {
public:
  /** Constructor
   * @param index: integer index of the wavevector in the Points object.
   * @param umklappVector: the crystal coordinates of a possible umklapp
   * vector, used for example in sum or differences between wavevectors.
   * @param points: the points object that this Point object belongs to.
   */
  Point(Points &points_, const int &index_,
        const Eigen::Vector3d &umklappVector = Eigen::Vector3d::Zero());

  /** copy constructor
   */
  Point(const Point &that);

  /** copy assignment operator
   */
  Point &operator=(const Point &that);

  /** Get the coordinates of the k-point
   * @param basis: either "cartesian" or "crystal"
   * @param inWignerSeitz: default false, if true, folds point in WS cell.
   * @return coordinates: a 3d vector of coordinates
   */
  Eigen::Vector3d getCoordinates(const int &basis = crystalCoordinates_,
                                 const bool &inWignerSeitz = false);

  /** Sum of two wavevectors (this + b)
   * The vector is folded in the Wigner Seitz zone with an Umklapp vector.
   */
  Point operator+(Point &b);

  /** Difference of two wavevectors (this-b).
   * The vector is folded in the Wigner Seitz zone with an Umklapp vector.
   */
  Point operator-(Point &b);

  /** checks whether the Point has been constructed with an Umklapp vector
   */
  bool hasUmklapp();

  /** Get the index of the wavevector in the referenced Points object.
   */
  int getIndex() const;

private:
  Eigen::Vector3d umklappVector;
  int index;
  Points &points;
};

/** Base class for storing wavevectors in the Brillouin zone.
 * The base class mostly refers to a uniform grid sampling the Brillouin zone.
 * Different subclasses will specialize to different kinds of wavevector sets.
 */
class Points {
public:
  /** Default constructor for a Monkhorst-Pack grid of wavevectors.
   * @param crystal: the crystal object that defines the Brillouin zone.
   * @param mesh: grid size of the Monkhorst-pack.
   * @param offset: the offset of the grid w.r.t. the Gamma point. Offset
   * ranges in values from 0 to 1. 0.5 means half displacement of the grid.
   */
  Points(Crystal &crystalObj_, const Eigen::Vector3i &mesh_,
         const Eigen::Vector3d &offset_ = Eigen::Vector3d::Zero());

  /** Constructor for points on a path.
   * @param crystal: the crystal object that defines the Brillouin zone.
   * @param pathExtrema: extrema of segments defining a path of points.
   * PathExtrema has size (numSegments, 2, 3), where the first index spans
   * the segments of the path, 2 is the beginning and the last element of the
   * segment, and 3 is the cartesian coordinates of the extrema.
   * Point coordinates must be in crystal coordinates.
   * Example: if we want a path G->X->L, we have: 2 segments: G->X and X->L
   * pathExtrema(0,0,:) = G. coordinates
   * pathExtrema(0,1,:) = X. coordinates
   * pathExtrema(1,0,:) = X. coordinates
   * pathExtrema(1,1,:) = L. coordinates
   * @param delta: segments are populated with points, with a distance
   * "delta" between different points in crystal coordinates.
   */
  Points(Crystal &crystal_, const Eigen::Tensor<double, 3> &pathExtrema,
         const double &delta);

  /** Constructor for Active Points
   * @param filter: a vector of integers of "filtered" points, i.e. the
   * indices of the wavevectors in parentPoints that we want to keep.
   */
  void setActiveLayer(const Eigen::VectorXi &filter_);

  /** Copy constructor
   */
  Points(const Points &obj);

  /** Copy assignment operator
   */
  Points &operator=(const Points &obj);

  /** Returns the details of the Monkhorst-pack mesh of wavevectors.
   * @return <grid, offset>: grid is the 3D integration mesh used for the
   * Brillouin zone sampling, and the 3D offset of the grid w.r.t. the
   * Gamma point (values of offset from 0 to 1).
   */
  std::tuple<Eigen::Vector3i, Eigen::Vector3d> getMesh();

  /** Returns the number of wavevectors stored in the points class.
   */
  int getNumPoints() const;

  /** Converts a wavevector from crystal to cartesian coordinates.
   * @param point: the input wavevector in crystal coordinates.
   * @return wavevector: the output wavevector in cartesian coordinates.
   */
  Eigen::Vector3d crystalToCartesian(const Eigen::Vector3d &point);

  /** Converts a wavevector from cartesian to crystal coordinates.
   * @param point: the input wavevector in cartesian coordinates.
   * @return wavevector: the output wavevector in crystal coordinates.
   */
  Eigen::Vector3d cartesianToCrystal(const Eigen::Vector3d &point);

  /** Folds a wavevector in crystal coordinates to the Wigner Seitz zone.
   * @param pointCrystal: the coordinates of a wavevector
   * @param basis: basis (Points::cartesianCoordinates or
   * Points::crystalCoordinates) of input and output wavevector.
   * @return wavevector: the wavevector coordinates folded in the WS zone.
   */
  Eigen::Vector3d bzToWs(const Eigen::Vector3d &point, const int &basis);

  Eigen::Vector3d foldToBz(const Eigen::Vector3d &pointCrystal,
                           const int &basis);

  /** Given a list of points, finds the monkhorst-pack mesh.
   * @param points: a matrix (3,numPoints) of wavevectors in crystal
   * coordinates.
   * @return mesh,offset: the monkhorst-pack grid size and the offset w.r.t.
   * the gamma point, that generates the input points list.
   * Calls an error if the mesh doesn't correspond to a complete list.
   */
  static std::tuple<Eigen::Vector3i, Eigen::Vector3d>
  findMesh(const Eigen::Matrix<double, 3, Eigen::Dynamic> &points);

  /** Returns a reference to the crystal object on which the Points are
   * defined.
   */
  Crystal &getCrystal();

  /** Returns a Point object given its integer index.
   * The Point object is used to move around the code the coordinates of the
   * wavevector.
   * @param index: the integer wavevector index ranging in [0,numPoints[
   * @return point: a point object.
   */
  Point getPoint(const int &index);

  /** Get the coordinates of a wavevector from its index.
   * @param index: the index of the desired wavevector.
   * @param basis: specify the basis to be used for the output coordinates.
   * Either Points::crystalCoordinates or Points::cartesianCoordinates.
   * @return wavevector: the coordinates of the desired wavevector.
   */
  Eigen::Vector3d getPointCoordinates(const int &index,
                                      const int &basis = crystalCoordinates);

  /** Get the wavevector index given the crystal coordinates of a wavevector.
   * Throws an error if the wavevector is not found in points.
   * @param point: the wavevector in crystal coordinates.
   * @return index: the index of the wavevector in the range [0,numPoints[
   */
  int getIndex(const Eigen::Vector3d &point);

  /** Like getIndex, get the wavevector index given the crystal coordinates
   * of a wavevector, but returns -1 if point not found.
   * @param crystalCoordinates_: a 3d Eigen vector with crystal coordinates of
   * a wavevector.
   * @return index: the index of the wavevector in the range [0,numPoints[ if
   * the point is in the set of points, or -1 if not found.
   */
  int isPointStored(const Eigen::Vector3d &crystalCoordinates_);

  // note: we could use constexpr to tell the compiler that the class member is
  // available at compilation time. But it wouldn't be compatible on old
  // clusters
  static const int crystalCoordinates;
  static const int cartesianCoordinates;

  /** setIrreduciblePoints analyzes a current set of points to find its
   * irreducible set. It also builds the logic for storing all symmetry
   * operations, such as finding which rotation maps an reducible point to its
   * irreducible point and similar symmetry operations.
   *
   * @param groupVelocities: an optional parameter, one can pass the values of
   * the group velocities at all wavevectors. If passed, setIrreduciblePoints
   * will try to find the best set of symmetries acting on wavevector that
   * also transforms the group velocities (i.e. there may sometimes be more than
   * one symmetry mapping v and k to irreducible values). Useful since we want
   * symmetries in the BTE to be the same as group velocities.
   */
  void
  setIrreduciblePoints(std::vector<Eigen::MatrixXd> *groupVelocities = nullptr,
                       std::vector<Eigen::VectorXd> *energies = nullptr);

  /** Returns an iterator on the index of irreducible points. If
   * setIrreduciblePoints has not been called, it just loops on all points.
   *
   * @return std::vector<int>: a list of irreducible wavevector indices.
   */
  std::vector<int> irrPointsIterator();

  /** Similar to irrPointsIterator, returns an iterator on the index of
   * irreducible points. If setIrreduciblePoints has not been called, it just
   * loops on all points. However, the returned list has already been scattered
   * across different MPI processes, so each MPI process has only a subset of
   * irreducible points.
   *
   * Note that symmetries do introduce some load unbalance. In fact, some
   * irreducible points may have more symmetry-equivalent points than others,
   * and thus may perform more work.
   *
   * @return std::vector<int> a MPI-distributed list of irreducible wavevector
   * indices.
   */
  std::vector<int> parallelIrrPointsIterator();

  /** Given an index of an irreducible point in the set [0,numPoints[, returns
   * its index in the set of [0,numIrredPoints[ irreducible wavevectors.
   *
   * @param ik: index of the irreducible point in the reducible list.
   * @return ikIrr: index in the irreducible list.
   */
  int asIrreducibleIndex(const int &ik);

  /** Given the index of an irreducible point in the set [0,numIrredPoints[,
   * returns its index in the set of [0,numPoints[ reducible wavevectors.
   *
   * @param ik: index of the irreducible point in the irreducible list.
   * @return ikIrr: index in the reducible list.
   */
  int asReducibleIndex(const int &ik);

  /** Given the coordinates of a wavevector, getRotationToIrreducible returns
   * the index of the irreducible point, and the rotation R such that
   * k^irr = R k^red
   *
   * @param x: a 3d vector with the coordinates of the wavevector. They can
   * be in crystal or cartesian basis, as specified by basis.
   * @param basis: either Points::cartesianCoordinates or crystalCoordinates,
   * specifies the basis in which x is specified.
   * @return tuple: the first element of the tuple is the index of the
   * irreducible point, the second is the 3d matrix with the rotation symmetry
   * operation. The rotation is crystal or cartesian, depending on the value of
   * basis.
   */
  std::tuple<int, Eigen::Matrix3d>
  getRotationToIrreducible(const Eigen::Vector3d &x,
                           const int &basis = cartesianCoordinates);

  /** Given the index of an irreducible point in the reducible list, returns the
   * set of rotations that allow to map k^red = R k^irr.
   *
   * @param ik: index of irreducible point in the reducible list.
   * @return vector<Matrix3d>: a vector with the set of rotation to reconstruct
   * the star.
   */
  std::vector<Eigen::Matrix3d> getRotationsStar(const int &ik);

  /** Given the index of an irreducible point in the reducible list, returns the
  * list of indices of the symmetry equivalent wavevectors.
  *
  * @param ik: index of irreducible point in the reducible list.
  * @return vector<int>: a vector with the indices of symmetry-equivalent points
   */
  std::vector<int> getReducibleStarFromIrreducible(const int &ik);

  /** Change the crystal object for a different one. Used as a helper by the
  * symmetrize function in the bandstructure class.
  * @param crystal: the new crystal to swap in
  */
  void swapCrystal(Crystal &newCrystal);

  /** getRotationFromReducibleIndex does the following.
   * Given the index of a point in the full list, it returns the rotation "rot"
   * that maps the reducible point to the irreducible one, such that
   * vRed = rot * vIrr , where v* is a vector with the crystal symmetries.
   *
   * @param ikFull: index of k-point in the full list of points.
   * @return rot: the rotation mapping the irreducible point to the reducible
   * point ik.
   */
  Eigen::Matrix3d getRotationFromReducibleIndex(int ikFull);

protected:
  void setMesh(const Eigen::Vector3i &mesh_, const Eigen::Vector3d &offset_);
  Crystal *crystalObj;
  Eigen::Vector3i mesh;
  Eigen::Vector3d offset;
  int numPoints = 0;
  // for Wigner Seitz folding
  Eigen::MatrixXd gVectors;

  // to store points on a path or active points
  bool explicitlyStored = false;
  bool isPointsListSorted = false;
  Eigen::MatrixXd pointsList;

  // for active Points:
  Eigen::VectorXi filteredToFullIndices;

  void setupGVectors();

  //------------------------------------------
  // Symmetries

  std::vector<Eigen::Matrix3d> rotationMatricesCrystal;
  std::vector<Eigen::Matrix3d> rotationMatricesCartesian;

  // vector of size 0 to numPoints: given index i, mERI(i) is the index of
  // the rotation R such that R*kIrr = k(i)
  Eigen::VectorXi mapEquivalenceRotationIndex;

  // vector of size 0 to numIrrPoints: given an irreducible wavevector i,
  // where i runs from 0 to numIrrPoints,
  // mapIrreducibleToReducibleList(i) is the index of the irreducible point in
  // the reducible list
  Eigen::VectorXi mapIrreducibleToReducibleList;

  // vector of size 0 to numPoints. Given a reducible wavevector i,
  // mRTI
  Eigen::VectorXi mapReducibleToIrreducibleList;
  int numIrrPoints = 0;

  // vector of size numIrrPoints. Given an irreducible point i
  // (i from 0,numIrrPoints), gets the list of reducible point indices
  // that are symmetry equivalent to point i.
  std::vector<std::vector<int>> irreducibleStars;

  // if equiv(i) == i, point is irreducible, otherwise, equiv(i) gives the
  // index of the irreducible equivalent index.
  Eigen::VectorXi equiv;
  //------------------------------------------
};

#endif
