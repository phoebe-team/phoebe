#include "points.h"
#include "eigen.h"
#include "mpiHelper.h"
#include "utilities.h"
#include <cmath>
#include <set>

const int Points::crystalCoordinates = crystalCoordinates_;
const int Points::cartesianCoordinates = cartesianCoordinates_;

// constructors
Points::Points(Crystal &crystalObj_, const Eigen::Vector3i &mesh_, const Eigen::Vector3d &offset_)
    : crystalObj{crystalObj_} {

  setMesh(mesh_, offset_);
  setupGVectors();
}

// default constructor
Points::Points(Crystal &crystalObj_) : crystalObj{crystalObj_} {
}

void Points::setupGVectors() {
  // This block allows the folding to the wigner seitz zone
  int nGx = 2;
  int nGVectors = (2 * nGx + 1) * (2 * nGx + 1) * (2 * nGx + 1);
  gVectors = Eigen::MatrixXd::Zero(3, nGVectors);
  Eigen::Matrix3d reciprocalUnitCell = crystalObj.getReciprocalUnitCell();
  nGVectors = 1; // we skip the first point which is G=(0,0,0)
  for (int i1 = -nGx; i1 <= nGx; i1++) {
    for (int i2 = -nGx; i2 <= nGx; i2++) {
      for (int i3 = -nGx; i3 <= nGx; i3++) {
        if ((i1 <= 0) || (i2 <= 0) || (i3 <= 0)) {
          Eigen::Vector3d vec;
          vec << i1, i2, i3;
          gVectors.col(nGVectors) = reciprocalUnitCell * vec;
          ++nGVectors;
        }
      }
    }
  }
}

Points::Points(Crystal &crystal_, const Eigen::Tensor<double, 3> &pathExtrema,
               const double &delta)
    : crystalObj(crystal_) {

  const double epsilon8 = 1.0e-10;

  explicitlyStored = true;

  setupGVectors();

  // build the list of points
  std::vector<Eigen::Vector3d> points;
  Eigen::Vector3d p0, p1; //, thisDelta;

  // initialize the path
  p0(0) = pathExtrema(0, 0, 0);
  p0(1) = pathExtrema(0, 0, 1);
  p0(2) = pathExtrema(0, 0, 2);
  points.push_back(p0);

  // we loop over the segments provided in user input
  for (int i = 0; i < pathExtrema.dimension(0); i++) {
    // load coordinates of the extrema of the segment
    p0(0) = pathExtrema(i, 0, 0);
    p0(1) = pathExtrema(i, 0, 1);
    p0(2) = pathExtrema(i, 0, 2);
    p1(0) = pathExtrema(i, 1, 0);
    p1(1) = pathExtrema(i, 1, 1);
    p1(2) = pathExtrema(i, 1, 2);

    // delta may not divide the interval exactly
    // so, we find the closest one
    int nk = abs(int((p1 - p0).norm() / delta));

    // now we build the points of the segment
    std::vector<Eigen::Vector3d> segmentPoints;
    for (int j = 0; j <= nk; j++) {
      Eigen::Vector3d thisP;
      thisP(0) = (p1(0) - p0(0)) / nk * j + p0(0);
      thisP(1) = (p1(1) - p0(1)) / nk * j + p0(1);
      thisP(2) = (p1(2) - p0(2)) / nk * j + p0(2);
      segmentPoints.push_back(thisP);
    }

    // now we update the original list of points
    // but we have to check to not add the same extrema twice
    // work with the first element
    Eigen::Vector3d lastP = points[points.size() - 1];
    Eigen::Vector3d thisP = segmentPoints[0];
    bool addIt = (thisP - lastP).norm() > epsilon8;
    if (addIt) {
      points.push_back(thisP);
    }

    // select first element of the segment
    auto p = std::begin(segmentPoints);
    ++p;
    // and then add all the rest of the segment
    for (auto end = std::end(segmentPoints); p != end; ++p) {
      // iterate over the rest of the container
      points.push_back(*p);
    }
  }

  numPoints = int(points.size());
  pointsList = Eigen::MatrixXd::Zero(3, numPoints);
  isPointsListSorted = true; // presume points are sorted
  int i = 0;
  for (const auto &p : points) {
    pointsList.col(i) = p;
    if ( i > 0) { // check list is sorted
      Eigen::Vector3d pOld = points[i-1];
      if ( p(0)<=pOld(0) && p(1)<=pOld(1) && p(2)<=pOld(2) ) {
        isPointsListSorted = false;
      }
    }
    ++i;
  }

}


/*void Points::swapCrystal(Crystal &newCrystal) {
  crystalObj = newCrystal;
  rotationMatricesCrystal.resize(0);
  rotationMatricesCartesian.resize(0);
  mapEquivalenceRotationIndex.resize(0);
  mapIrreducibleToReducibleList.resize(0);
  mapReducibleToIrreducibleList.resize(0);
  numIrrPoints = 0;
  irreducibleStars.resize(0);
  equiv.resize(0);
}*/

void Points::setActiveLayer(const Eigen::VectorXi &filter) {
  // if the filter has the same size of points, we are not filtering anything
  // and we just use the class as a full list of points
  if (filter.size() == numPoints) {
    return;
  }

  // this contain the list of indices of the points in the FullPoints class
  // which we want to include in the ActivePoints class
  filteredToFullIndices = filter;

  numPoints = int(filteredToFullIndices.size());

  int maxIndex = 0;

  isPointsListSorted = true; // presume points are sorted
  // we then construct the list of points
  pointsList.resize(3, numPoints);
  pointsList.setZero();
  for (int ikNew = 0; ikNew < numPoints; ikNew++) {
    int ik = filteredToFullIndices(ikNew);
    Eigen::Vector3d x = getPointCoordinates(ik, Points::crystalCoordinates);
    pointsList.col(ikNew) = x;

    if (ik > maxIndex)
      ik = maxIndex;
    if (ik < 0)
      Error("Negative point filter is not valid");

    if ( ikNew > 0) { // check list is sorted
      Eigen::Vector3d pOld = pointsList.col(ikNew-1);
      if ( x(0)<=pOld(0) && x(1)<=pOld(1) && x(2)<=pOld(2) ) {
        isPointsListSorted = false;
      }
    }
  }

  if (maxIndex + 1 > mesh.prod()) {
    Error("Filter doesn't run on the list of points");
  }

  // note: we place this after building pointsList,
  // because this changes the behavior of getCoordinates
  explicitlyStored = true;

  if (numIrrPoints > 0) {
    Error("Setting symmetries before applying active layer is unsupported");
    // Note: it's not impossible, we simply didn't implement it because not
    // needed at the time of writing this
  }
}

// copy constructor
Points::Points(const Points &that)
    : crystalObj(that.crystalObj), mesh(that.mesh), offset(that.offset),
      numPoints(that.numPoints), gVectors(that.gVectors),
      explicitlyStored(that.explicitlyStored),
      isPointsListSorted(that.isPointsListSorted),
      pointsList(that.pointsList),
      filteredToFullIndices(that.filteredToFullIndices),
      rotationMatricesCrystal(that.rotationMatricesCrystal),
      rotationMatricesCartesian(that.rotationMatricesCartesian),
      mapEquivalenceRotationIndex(that.mapEquivalenceRotationIndex),
      mapIrreducibleToReducibleList(that.mapIrreducibleToReducibleList),
      mapReducibleToIrreducibleList(that.mapReducibleToIrreducibleList),
      numIrrPoints(that.numIrrPoints), irreducibleStars(that.irreducibleStars),
      equiv(that.equiv) {}

// copy assignment operator
Points &Points::operator=(const Points &that) { // assignment operator
  if (this != &that) {
    crystalObj = that.crystalObj;
    mesh = that.mesh;
    offset = that.offset;
    numPoints = that.numPoints;
    gVectors = that.gVectors;
    explicitlyStored = that.explicitlyStored;
    isPointsListSorted = that.isPointsListSorted;
    pointsList = that.pointsList;
    filteredToFullIndices = that.filteredToFullIndices;
    rotationMatricesCrystal = that.rotationMatricesCrystal;
    rotationMatricesCartesian = that.rotationMatricesCartesian;
    mapEquivalenceRotationIndex = that.mapEquivalenceRotationIndex;
    mapIrreducibleToReducibleList = that.mapIrreducibleToReducibleList;
    mapReducibleToIrreducibleList = that.mapReducibleToIrreducibleList;
    numIrrPoints = that.numIrrPoints;
    irreducibleStars = that.irreducibleStars;
    equiv = that.equiv;
  }
  return *this;
}

Point Points::getPoint(const int &index) { return Point(*this, index); }

// getPointCoordinates is the tool to find the coordinates of a point

Eigen::Vector3d Points::getPointCoordinates(const int &index,
                                            const int &basis) {
  if (basis != crystalCoordinates && basis != cartesianCoordinates) {
    Error("Wrong basis for getPoint");
  }
  Eigen::Vector3d pointCrystal;
  if (explicitlyStored) {
    // the coordinates are stored, we just retrieve them
    pointCrystal = pointsList.col(index);
  } else {
    // in this case we have a mesh
    int ikz = index / (mesh(0) * mesh(1));
    int idx_ = index - (ikz * mesh(0) * mesh(1));
    int iky = idx_ / mesh(0);
    int ikx = mod(idx_, int(mesh(0)));
    Eigen::Vector3d p;
    pointCrystal(0) = double(ikx) / (double)mesh(0) + offset(0);
    pointCrystal(1) = double(iky) / (double)mesh(1) + offset(1);
    pointCrystal(2) = double(ikz) / (double)mesh(2) + offset(2);
  }
  if (basis == crystalCoordinates) {
    return pointCrystal;
  } else {
    Eigen::Vector3d pointCartesian = crystalToCartesian(pointCrystal);
    return pointCartesian;
  }
}

int Points::getIndex(const Eigen::Vector3d &point) {
  int ik = isPointStored(point);
  if (ik == -1) {
    Error("getIndex found a point not falling on the wavevector mesh");
  }
  return ik;
}

int Point::getIndex() const { return index; }

/** This is the plain binary search algorithm.
 * Algorithm scaling of log(numPoints)
 *
 * @param pointsList: sorted list of wavevectors
 * @param left: leftr- lowest index of the search interval
 * @param right: right- uppermost index of the search interval
 * @param x: wavevector to be found
 * @param direction: cartesian direction currently under exam
 * @return -1 if point not found, or the index of the point in the list.
 */
int innermostPointsBinarySearch(const Eigen::MatrixXd& pointsList,
                                const int& left, const int& right,
                                const Eigen::Vector3d& x,
                                const int& direction) {
  if (right >= left) {
    int mid = left + (right - left) / 2;

    // If the element is present at the middle itself
    double diff = pointsList(direction,mid) - x(direction);

    if ( abs(diff) < 1.0e-8 ) {
      return mid;
    }

    // If element is smaller than mid, then it can only be in left subarray
    if ( diff > 1.0e-8 ) {
      return innermostPointsBinarySearch(pointsList, left, mid - 1, x, direction);
    }

    // Else the element can only be present in right subarray
    return innermostPointsBinarySearch(pointsList, mid + 1, right, x, direction);
  }

  // We reach here when element is not present in array
  return -1;
}

/** Binary search algorithm for a fast point lookup.
 * Since the binary search doesn't check for duplicates, here we use the index
 * found by the binary search algorithm, and move to it's right or left to check
 * if the point has the same coordinates on the directions that have been
 * searched.
 * Total cost is log(n+k) where n is number of points being searched, and k the
 * number of duplicates.
 *
 * @param pointsList: (3,nk) matrix of wavevector coordinates
 * @param left: left index value. Initialize it with 0
 * @param right: right index value. Initialize it with Nk-1
 * @param x: coordinates of the point to be found.
 * @param direction: [0,1,2] the cartesian direction to search
 * @return: the index of a point or -1 if point is not present.
 */
std::pair<int,int> internalPointsBinarySearch(const Eigen::MatrixXd& pointsList,
                                              const int& idMin, const int&idMax,
                                              const Eigen::Vector3d& point,
                                              const int& direction) {
  // first we start with the z coordinate, looking for matching values.
  int id = innermostPointsBinarySearch(pointsList, idMin, idMax, point,
                                       direction);

  if (id == -1) { // point was not found
    return std::make_pair(-1, -1);
  }

  // note: the binary search per-se requires unique elements
  // points list is unique on the vector, but single components are duplicated.
  // so, we look for the first and last matching component
  int idxMin = id;
  while (idxMin > 0) { // stop if idx reaches 0
    // we also need to check the other components already checked are fine

    double diff = 0.;
    for (int i=direction; i<3; i++) {
      double diff2 = pointsList(i, idxMin - 1) - point(i);
      diff += diff2 * diff2;
    }

    // if still the same point at a lower index, save
    if ( abs(diff) < 1.0e-8) {
      idxMin -= 1;
    } else {
      break; // exit if point is different
    }
  }
  int idxMax = id;
  while (idxMax < pointsList.cols()-1) { // stop if idx reaches 0
    double diff = 0.;
    for (int i=direction; i<3; i++) {
      double diff2 = pointsList(i, idxMax + 1) - point(i);
      diff += diff2 * diff2;
    }
    // if still the same point at a lower index, save
    if ( abs(diff) < 1.0e-8) {
      idxMax += 1;
    } else {
      break; // exit if point is different
    }
  }
  return std::make_pair(idxMin, idxMax);
}

/** Binary search algorithm to look for a vector in a list of vector.
 * It only makes sense if the list is sorted by z, then y, and x coordinates,
 * in this precise order. If duplicate points are found, an error is thrown.
 * In this function, we first search through z to restrict the indices, then
 * further restrict indices by comparing y and x coordinate.
 *
 * @param pointsList: list of vectors, in crystal coordinates.
 * @param point: crystal coordinates of the vector to be found.
 * @return idx: the index of the point in the list. -1 if not found.
 */
int pointsBinarySearch(const Eigen::MatrixXd& pointsList,
                       const Eigen::Vector3d& point) {
  // we do first z then y and x coordinate. This must be kept consistent in the
  // points generation above, and also windows must preserve the order.

  // start with the z coordinate
  int start = 0;
  int stop = pointsList.cols() - 1;
  auto tz = internalPointsBinarySearch(pointsList, start, stop, point, 2);
  int idzMin = std::get<0>(tz);
  int idzMax = std::get<1>(tz);

  if (idzMin == -1 || idzMax == -1) return -1; // point not found

  // now the y coordinate, restricting the list to the right z coordinates.
  auto ty = internalPointsBinarySearch(pointsList, idzMin, idzMax, point, 1);
  int idyMin = std::get<0>(ty);
  int idyMax = std::get<1>(ty);
  if (idyMin == -1 || idyMax == -1) return -1; // point not found

  // now the y coordinate, restricting the list to the right z coordinates.
  auto tx = internalPointsBinarySearch(pointsList, idyMin, idyMax, point, 0);
  int idxMin = std::get<0>(tx);
  int idxMax = std::get<1>(tx);
  if (idxMin == -1 || idxMax == -1) return -1; // point not found

  if ( idxMin != idxMax ) {
    Error("Duplicate points found");
  }

  return idxMin; // point found
}

int Points::isPointStored(const Eigen::Vector3d &crystCoordinates_) {

  if (!explicitlyStored) { // full list is faster
    Eigen::Vector3i p;
    // multiply by grid, so that p now contains integers
    double diff = 0.;
    for (int i : {0, 1, 2}) {
      // bring the point to integer coordinates
      double x = (crystCoordinates_(i) - offset(i)) * mesh(i);
      // check that p is indeed a point commensurate to the mesh.
      diff += round(x) - x;
      // fold in Brillouin zone in range [0,mesh-1]
      p(i) = mod(int(round(x)), mesh(i));
    }
    if (diff >= 1.0e-6) {
      int ik = -1;
      return ik;
    }
    int ik = p(2) * mesh(0) * mesh(1) + p(1) * mesh(0) + p(0);
    return ik;
  } else {

    if ( isPointsListSorted ) {
      // since pointsList is a sorted list, we can use binary search to find the
      // point index in log(N) iterations

      // first, we fold the point in the interval [0,1[
      // the same for which the points are stored
      Eigen::Vector3d thePoint = crystCoordinates_;
      for ( int i : {0,1,2}) {
        thePoint(i) -= std::floor(thePoint(i) + 1.0e-8);
      }
      int ik = pointsBinarySearch(pointsList, thePoint);
      return ik;
    } else { // if not sorted, we just do a search over the full list
      int ik = -1;
      for (int ikTest = 0; ikTest < numPoints; ikTest++) {
        Eigen::Vector3d kTest = pointsList.col(ikTest);
        Eigen::Vector3d diff = kTest - crystCoordinates_;
        for (int i : {0, 1, 2}) {
          diff(i) -= std::floor(diff(i) + 1.0e-8);
        }
        if (diff.squaredNorm() < 1.0e-5) {
          ik = ikTest;
        }
      }
      return ik;
    }
  }
}

// change of basis methods

Eigen::Vector3d Points::crystalToCartesian(const Eigen::Vector3d &point) {
  return crystalObj.getReciprocalUnitCell() * point;
}

Eigen::Vector3d Points::cartesianToCrystal(const Eigen::Vector3d &point) {
  Eigen::Vector3d p = crystalObj.getReciprocalUnitCell().inverse() * point;
  return p;
}

Eigen::Vector3d Points::bzToWs(const Eigen::Vector3d &point, const int &basis) {

  Eigen::Vector3d pointCrystal = point;
  if (basis == cartesianCoordinates) {
    pointCrystal = cartesianToCrystal(pointCrystal);
  }

  // fold to BZ first
  pointCrystal = foldToBz(pointCrystal, crystalCoordinates);

  Eigen::Vector3d pointCart = crystalToCartesian(pointCrystal);

  double norm2 = pointCart.squaredNorm();
  int iws = 0;
  Eigen::Vector3d constShift;
  constShift.setConstant(1.0e-8);
  for (int iG = 1; iG < gVectors.cols(); iG++) {
    Eigen::Vector3d thisG = gVectors.col(iG);
    double thisNorm2 = (pointCart + thisG + constShift).squaredNorm();
    if (thisNorm2 < norm2) {
      norm2 = thisNorm2;
      iws = iG;
    }
  }

  Eigen::Vector3d point2 = pointCart + gVectors.col(iws);
  if (basis == crystalCoordinates) {
    point2 = cartesianToCrystal(point2);
  }
  return point2;
}

Eigen::Vector3d Points::foldToBz(const Eigen::Vector3d &point,
                                 const int &basis) {
  if (basis != crystalCoordinates && basis != cartesianCoordinates) {
    Error("Wrong input to Wigner Seitz folding");
  }

  Eigen::Vector3d pointCrystal = point;
  if (basis == cartesianCoordinates) {
    pointCrystal = cartesianToCrystal(pointCrystal);
  }

  for (int i : {0, 1, 2}) {
    // note: the small double added takes into account for rounding errors
    // which may arise e.g. when the points come from rotations, or machine
    // precision errors in the conversion cartesian to crystal
    // is only a problem in the (unlikely) case of huge k-point meshes (10^8)
    pointCrystal(i) -= std::floor(pointCrystal(i) + 1.0e-8);
  }

  Eigen::Vector3d point2 = pointCrystal;
  if (basis == cartesianCoordinates) {
    point2 = crystalToCartesian(point2);
  }
  return point2;
}

Crystal &Points::getCrystal() { return crystalObj; }

int Points::getNumPoints() const { return numPoints; }

void Points::setMesh(const Eigen::Vector3i &mesh_,
                     const Eigen::Vector3d &offset_) {

  // validate the mesh and then store it
  if (mesh_(0) <= 0) {
    Error("meshGrid(0) <= 0, should be positive");
  }
  if (mesh_(1) <= 0) {
    Error("meshGrid(1) <= 0, should be positive");
  }
  if (mesh_(2) <= 0) {
    Error("meshGrid(2) <= 0, should be positive");
  }
  mesh = mesh_;
  numPoints = mesh(0) * mesh(1) * mesh(2);

  if (offset_(0) < 0. && offset_(0) >= 1.) {
    Error("offset(0) should be 0 <= offset < 1");
  }
  if (offset_(1) < 0. && offset_(1) >= 1.) {
    Error("offset(1) should be 0 <= offset < 1");
  }
  if (offset_(2) < 0. && offset_(2) >= 1.) {
    Error("offset(2) should be 0 <= offset < 1");
  }
  offset = offset_;

  // I won't set the list of reducible BZ points
  // we generate it on the fly
}

std::tuple<Eigen::Vector3i, Eigen::Vector3d> Points::getMesh() {
  return std::make_tuple(mesh, offset);
}

std::tuple<Eigen::Vector3i, Eigen::Vector3d>
Points::findMesh(const Eigen::Matrix<double, 3, Eigen::Dynamic> &testPoints) {
  // given a list of points, figures out the mesh and offset
  // input points must be in crystal coordinates
  Eigen::Vector3i mesh_(3);
  mesh_.setZero();
  Eigen::Vector3d offset_(3);
  offset_.setZero();

  auto numTestPoints = int(testPoints.cols());
  for (int iCart = 0; iCart < 3; iCart++) {
    std::set<double> s; // note that sets are ordered
    for (int i = 0; i < numTestPoints; i++) {
      double value = testPoints(iCart, i);
      // fold number in [0,1[
      while (value < 0.) {
        value += 1.;
      }
      while (value >= 1.) {
        value -= 1.;
      }
      s.insert(value);
    }

    // a few things to remember for comparison.
    // * there might be cases where deltaK is a periodic number (e.g. 1./6)
    // * we are working with floats, so there can be rounding errors
    //   and thus s might contain duplicates

    // determine if there is a single point
    bool isSingle = true;
    for (auto it = s.begin(); it != s.end(); it++) {
      if (it != s.begin()) { // if not in the first case
        // the first condition is the trivial check on having a
        // different point. The second condition is to exclude cases
        // such that s = [0., 0.99999]
        if ((*it - *s.begin() > 1.0e-6) && (1. - *it + *s.begin() > 1.0e-6)) {
          isSingle = false;
          break;
        }
      }
    }

    if (isSingle) { // no mesh, just a single point
      mesh_(iCart) = 1;
      offset_(iCart) = *s.begin();
    } else {
      double delta = 0.;
      for (auto it = s.begin(); it != s.end(); it++) {
        if (*it - *s.begin() > 1.0e-6) { // avoid duplicates
          delta = *it - *s.begin();
          break;
        }
      }
      mesh_(iCart) = int(round(1. / delta + 0.1));
      offset_(iCart) = *s.begin();
    }
  }

  if (numTestPoints != mesh_(0) * mesh_(1) * mesh_(2)) {
    Error("Mesh of points seems incomplete");
  }
  return std::make_tuple(mesh_, offset_);
}

////////////////////////////////////////////////////////////////

Point::Point(Points &points_, const int &index_, const Eigen::Vector3d &umklappVector_)
    : umklappVector(umklappVector_), index(index_), points(points_) {}

// copy constructor
Point::Point(const Point &that)
    : umklappVector(that.umklappVector), index(that.index),
      points(that.points) {}

// copy assignment
Point &Point::operator=(const Point &that) {
  if (this != &that) {
    umklappVector = that.umklappVector;
    index = that.index;
    points = that.points;
  }
  return *this;
}

Eigen::Vector3d Point::getCoordinates(const int &basis,
                                      const bool &inWignerSeitz) {
  if ((basis != Points::cartesianCoordinates) &&
      (basis != Points::crystalCoordinates)) {
    Error("Point getCoordinates: basis must be crystal or cartesian");
  }

  Eigen::Vector3d coordinates = points.getPointCoordinates(index, basis);
  if (inWignerSeitz) {
    coordinates = points.bzToWs(coordinates, basis);
  }
  return coordinates;
}

bool Point::hasUmklapp() {
  if (abs(umklappVector.norm()) < 1.0e-8) {
    return false;
  } else {
    return true;
  }
}

Point Point::operator+(Point &b) {
  if (&b.points != &points) {
    Error("Points sum should refer to points of the same mesh");
  }
  Eigen::Vector3d coordinates = getCoordinates() + b.getCoordinates();
  int ik = points.getIndex(coordinates);
  Eigen::Vector3d umklappVector2 = points.getPointCoordinates(ik) - coordinates;
  Point p(points, ik, umklappVector2);
  return p;
}

Point Point::operator-(Point &b) {
  if (&b.points != &points) {
    Error("Points sum should refer to points of the same mesh");
  }
  Eigen::Vector3d coordinates = getCoordinates() - b.getCoordinates();
  int ik = points.getIndex(coordinates);
  Eigen::Vector3d umklappVector2 = points.getPointCoordinates(ik) - coordinates;
  Point p(points, ik, umklappVector2);
  return p;
}

void Points::setIrreduciblePoints(
    std::vector<Eigen::MatrixXd> *groupVelocities) {
  // go through the list of wavevectors and find the irreducible points

  const double epsilon = 1.0e-8;

  equiv.resize(numPoints);
  for (int i = 0; i < numPoints; i++) {
    equiv(i) = i;
  }
  // equiv(i) = i : k-point i is not equivalent to any other (irreducible)
  // equiv(i)!=i : k-point i is equivalent to k-point equiv(nk)

  std::vector<SymmetryOperation> symmetries =
      crystalObj.getSymmetryOperations();
  {
    rotationMatricesCrystal.resize(0);
    rotationMatricesCartesian.resize(0);
    Eigen::Matrix3d bg = crystalObj.getReciprocalUnitCell();
    for (const SymmetryOperation &symmetry : symmetries) {
      Eigen::Matrix3d rotation = symmetry.rotation;
      rotationMatricesCrystal.push_back(rotation);
      rotation = bg * rotation * bg.inverse();
      rotationMatricesCartesian.push_back(rotation);
    }
  }

  // the identity is assumed to be the first symmetry. Let's check it
  {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    double x = (rotationMatricesCrystal[0] - I).norm();
    if (x > epsilon) {
      Error("Development bug: Identity should be the first symmetry");
    }
  }

  // search for irreducible kpoints, checking every point
  for (int ik = 0; ik < numPoints; ik++) {
    // check if this k-point has already been found equivalent to another
    // if true, this point has not yet been identified as reducible
    if (equiv(ik) == ik) {

      // check if there are equivalent k-point to this in the list
      // (excepted those previously found to be equivalent to another)

      std::set<int> thisStar;

      // apply symmetries to identify all other points
      // which will reduce to this point, and save them to its star
      for (const auto &symmetry : symmetries) {
        Eigen::Matrix3d rot = symmetry.rotation;
        Eigen::Vector3d rotatedPoint =
            rot * getPointCoordinates(ik, Points::crystalCoordinates);

        // check if rotated point is somewhere on the mesh
        int ikRot = isPointStored(rotatedPoint);

        // if the point is on the mesh and is not previously identified
        // as an irreducible point
        if (ikRot >= 0 && equiv(ikRot) == ikRot) {
          if (ikRot >= ik) {
            // if the rotated point is further in the list than ik, denote ik as
            // its ikRot's irr point
            equiv(ikRot) = ik;
            thisStar.insert(ikRot);
          } else {
            // if ikRot was before ik in the list, yet we
            // somehow did not flag ikRot as ik's irr kpoint,
            // there has been an error
            if (equiv(ikRot) != ik || ikRot < ik) {
              Error("Error in finding irreducible points");
            }
          }
        }
      }

      std::vector<int> tmp;
      std::copy(thisStar.begin(), thisStar.end(), std::back_inserter(tmp));
      irreducibleStars.push_back(tmp); // here we save the equivalent star
    }
  }

  // here we find the rotations mapping the reducible point ot the irreducible
  // Note: I want to see how velocities rotate.
  mapEquivalenceRotationIndex = Eigen::VectorXi::Zero(numPoints);
  mapEquivalenceRotationIndex.setConstant(-1);
  if (groupVelocities == nullptr) {
    for (int ikRed = 0; ikRed < numPoints; ikRed++) {
      if (equiv(ikRed) == ikRed) {              // identity
        mapEquivalenceRotationIndex(ikRed) = 0; // checked above it's identity
      } else {
        int ikIrr = equiv(ikRed);

        Eigen::Vector3d kIrr =
            getPointCoordinates(ikIrr, Points::crystalCoordinates);

        for (unsigned int is = 0; is < symmetries.size(); is++) {
          Eigen::Matrix3d rot = rotationMatricesCrystal[is];
          Eigen::Vector3d rotatedPoint = rot * kIrr;
          int ikRot = isPointStored(rotatedPoint);
          if (ikRot == ikRed) {
            mapEquivalenceRotationIndex(ikRed) = int(is);
            break;
          }
        }
      }
    }
  } else {
    if (int((*groupVelocities).size()) != numPoints) {
      Error("setIrreducible: velocities not aligned with the full grid");
    }

    for (int ikRed = 0; ikRed < numPoints; ikRed++) {
      if (equiv(ikRed) == ikRed) {              // identity
        mapEquivalenceRotationIndex(ikRed) = 0; // checked above it's identity
      } else {
        int ikIrr = equiv(ikRed);

        Eigen::Vector3d kIrr =
            getPointCoordinates(ikIrr, Points::crystalCoordinates);

        Eigen::MatrixXd irrVelocities = (*groupVelocities)[ikIrr];
        Eigen::MatrixXd redVelocities = (*groupVelocities)[ikRed];

        if (irrVelocities.rows() != redVelocities.rows()) {
          Error("Different number of bands at two equivalent points");
        }

        std::vector<int> isSelects;
        std::vector<double> diffs;

        for (unsigned int is = 0; is < symmetries.size(); is++) {
          Eigen::Vector3d rotatedPoint = rotationMatricesCrystal[is] * kIrr;
          int ikRot = isPointStored(rotatedPoint);
          if (ikRot != ikRed) {
            continue;
          }

          auto numBands = int(irrVelocities.rows());
          double diff = 0.;
          for (int ib = 0; ib < numBands; ib++) {
            Eigen::Vector3d thisIrrVel = irrVelocities.row(ib);
            Eigen::Vector3d thisRedVel = redVelocities.row(ib);
            Eigen::Vector3d rotVel = rotationMatricesCartesian[is] * thisIrrVel;
            diff += (rotVel - thisRedVel).squaredNorm();
          }
          isSelects.push_back(int(is));
          diffs.push_back(diff);
        }

        auto minElementIndex =
            std::min_element(diffs.begin(), diffs.end()) - diffs.begin();

        mapEquivalenceRotationIndex(ikRed) = isSelects[minElementIndex];
      }
    }
  }

  // count number of irreducible points
  numIrrPoints = 0;
  for (int ik = 0; ik < numPoints; ik++) {
    if (equiv(ik) == ik) {
      ++numIrrPoints;
    }
  }

  // this allows us to map (ikIrr in the irrPoints) -> (ikRed in fullPoints)
  mapIrreducibleToReducibleList = Eigen::VectorXi::Zero(numIrrPoints);
  {
    int i = 0;
    for (int j = 0; j < numPoints; j++) {
      if (equiv(j) == j) { // if is irreducible
        mapIrreducibleToReducibleList(i) = j;
        ++i;
      }
    }
  }

  // this allows us to map (ikRed in fullPoints) -> (ikIrr in the irrPoints)
  // basically the inverse of mapIrrToRedList
  mapReducibleToIrreducibleList = Eigen::VectorXi::Zero(numPoints);

  // make a vector copy of mI2R for binary search
  std::vector<int> mI2R(mapIrreducibleToReducibleList.data(),
                   mapIrreducibleToReducibleList.data()
                        + mapIrreducibleToReducibleList.size());

  for (int ik : mpi->divideWorkIter(numPoints)) {
    int ikIrr = equiv(ik); // map to the irreducible in fullPoints
    auto itr = std::lower_bound(mI2R.begin(), mI2R.end(), ikIrr);
    int ikIrr2 = std::distance(mI2R.begin(), itr);
    if (ikIrr2 == -1) {
      Error("Failed building irreducible points mapRedToIrrList");
    }
    mapReducibleToIrreducibleList(ik) = ikIrr2;
  }
  mpi->allReduceSum(&mapReducibleToIrreducibleList);
}

std::vector<int> Points::irrPointsIterator() {
  std::vector<int> iter;
  if (numIrrPoints == 0) {
    for (int ik = 0; ik < numPoints; ik++) {
      iter.push_back(ik);
    }
  } else {
    for (int ik = 0; ik < numPoints; ik++) {
      if (equiv(ik) == ik) {
        iter.push_back(ik);
      }
    }
  }
  return iter;
}

std::vector<int> Points::parallelIrrPointsIterator() {
  auto v = irrPointsIterator();
  //
  auto divs = mpi->divideWork(v.size());
  int start = divs[0];
  int stop = divs[1];
  //
  std::vector<int> iter(v.begin() + start, v.begin() + stop);
  return iter;
}

int Points::asIrreducibleIndex(const int &ik) {
  if (numIrrPoints > 0) { // we set the irreducible points
    assert(ik < numPoints && ik >= 0);
    return mapReducibleToIrreducibleList(ik);
  } else { // no symmetries set, only the identity symmetry exists
    return ik;
  }
}

int Points::asReducibleIndex(const int &ik) {
  if (numIrrPoints > 0) { // we set the irreducible points
    assert(ik < numIrrPoints && ik >= 0);
    return mapIrreducibleToReducibleList(ik);
  } else { // no symmetries set, only the identity symmetry exists
    return ik;
  }
}

std::tuple<int, Eigen::Matrix3d>
Points::getRotationToIrreducible(const Eigen::Vector3d &x, const int &basis) {
  Eigen::Vector3d xCrystal;
  if (basis == cartesianCoordinates) {
    xCrystal = cartesianToCrystal(x);
  } else {
    xCrystal = x;
  }
  // find the index in the list
  int ik = getIndex(xCrystal);

  if (numIrrPoints > 0) { // if irreducible points have been set
    // find rotation such that rotation * qFull = qRed
    Eigen::Matrix3d rot;
    if (basis == crystalCoordinates) {
      rot = rotationMatricesCrystal[mapEquivalenceRotationIndex(ik)].inverse();
    } else {
      rot =
          rotationMatricesCartesian[mapEquivalenceRotationIndex(ik)].inverse();
    }
    // also, we add the index of the irreducible point to which x is mapped
    return std::make_tuple(equiv(ik), rot);
  } else {
    Eigen::Matrix3d identity;
    identity.setIdentity();
    return std::make_tuple(ik, identity);
  }
}

std::vector<Eigen::Matrix3d> Points::getRotationsStar(const int &ik) {
  if (numIrrPoints > 0) {
    // first, map ik to [0,N_irr]
    int ikIrr = mapReducibleToIrreducibleList(ik);
    std::vector<int> starIndices = irreducibleStars[ikIrr];
    std::vector<Eigen::Matrix3d> rotations;
    for (int ikFull : starIndices) {
      Eigen::Matrix3d rot;
      rot = rotationMatricesCartesian[mapEquivalenceRotationIndex(ikFull)];
      rotations.push_back(rot);
    }
    // list of rotations such that qStar = rotation * qIrr
    return rotations;
  } else {
    std::vector<Eigen::Matrix3d> rotations;
    Eigen::Matrix3d identity;
    identity.setIdentity();
    rotations.push_back(identity);
    return rotations;
  }
}

std::vector<int> Points::getReducibleStarFromIrreducible(const int &ik) {
  int ikIrr = mapReducibleToIrreducibleList(ik);
  return irreducibleStars[ikIrr];
}
