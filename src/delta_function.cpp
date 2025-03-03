#include "delta_function.h"
#include "constants.h"
#include "context.h"
#include "exceptions.h"
#include "utilities.h"

DeltaFunction::~DeltaFunction() {}

// functions to define the type
int DeltaFunction::getType() { return id; }
int GaussianDeltaFunction::getType() { return id; }
int AdaptiveGaussianDeltaFunction::getType() { return id; }
//int SymAdaptiveGaussianDeltaFunction::getType() { return id; }
int TetrahedronDeltaFunction::getType() { return id; }

// app factory
DeltaFunction * DeltaFunction::smearingFactory(Context &context,
                               BaseBandStructure &fullBandStructure) {

  auto choice = context.getSmearingMethod();
  if (choice == gaussian) {
    return new GaussianDeltaFunction(fullBandStructure, context);
  } else if (choice == adaptiveGaussian) {
    return new AdaptiveGaussianDeltaFunction(fullBandStructure, context);
  } else if (choice == tetrahedron) {
    return new TetrahedronDeltaFunction(fullBandStructure);
  } else if (choice == symAdaptiveGaussian) {
    return new SymAdaptiveGaussianDeltaFunction(fullBandStructure, context);
  } else {
    Error("Unrecognized smearing choice");
    return nullptr;
  }
}

// Gaussian smearing ----------------------------------------------------

GaussianDeltaFunction::GaussianDeltaFunction(BaseBandStructure& bandStructure, Context &context) {

  double smearingWidth = context.getSmearingWidth();

  if(!std::isnan(context.getPhSmearingWidth()) && bandStructure.getParticle().isPhonon() ) {
    smearingWidth = context.getPhSmearingWidth();
  }
  if(!std::isnan(context.getElSmearingWidth()) && bandStructure.getParticle().isElectron() ) {
    smearingWidth = context.getElSmearingWidth();
  }

  if(mpi->mpiHead()) {
    std::setprecision(9);
    std::cout << "\nGaussian smearing width is " << smearingWidth * 13.6057039763 << " eV." << std::endl;
  }

  inverseWidth = 1. / smearingWidth;
  prefactor = 1. / smearingWidth / sqrt(pi);

}

double GaussianDeltaFunction::getSmearing(const double &energy,
                                          [[maybe_unused]] const Eigen::Vector3d &velocity,
                                          [[maybe_unused]] const Eigen::Vector3d &velocity2,
                                          [[maybe_unused]] const Eigen::Vector3d &velocity3) {
  double x = energy * inverseWidth;
  if ( x > 6. ) return 0.; // the error is < 2e-16, and prevents underflow
  return prefactor * exp(-x * x);
}

double GaussianDeltaFunction::getSmearing([[maybe_unused]] const double &energy, [[maybe_unused]] StateIndex &is) {
  Error("GaussianDeltaFunction::getSmearing2 not implemented");
  return 1.;
}

// Adaptive gaussian smearing -----------------------------------------

AdaptiveGaussianDeltaFunction::AdaptiveGaussianDeltaFunction(
    BaseBandStructure &bandStructure, Context& context, double broadeningCutoff_) {
  auto tup = bandStructure.getPoints().getMesh();
  auto mesh = std::get<0>(tup);
  qTensor = bandStructure.getPoints().getCrystal().getReciprocalUnitCell();
  qTensor.row(0) /= mesh(0);
  qTensor.row(1) /= mesh(1);
  qTensor.row(2) /= mesh(2);
  broadeningCutoff = broadeningCutoff_;

  // if user set this input we override it here
  //if (!std::isnan(context.getAdaptiveSmearingPrefactor())) {

    //adaptivePrefactor = context.getAdaptiveSmearingPrefactor();
    //if(bandStructure.getParticle().isPhonon()) {
    //  adaptivePrefactor *= 250.; // bigger factor is better for ph it seems -- TODO this differently
    //}
/*     if(context.getAppName().find("Coupled") != std::string::npos) {
      Warning("For a coupled calculation, be careful about setting adaptiveSmearingPrefactor --\n"
		      "this will set the el and ph values to the same thing,\n"
		      "which is likely not optimal!");
    } */
  //}

  if(bandStructure.getParticle().isPhonon() ) {
    // set user defined smearing if requested
    if(!std::isnan(context.getPhSmearingWidth())) {
      adaptivePrefactor = context.getPhSmearingWidth();
    } else if (!std::isnan(context.getAdaptiveSmearingPrefactor())) {
      adaptivePrefactor = context.getAdaptiveSmearingPrefactor();
    } else {
      adaptivePrefactor = 0.05; // this is tested to work well for phonons
    }
  } else if (bandStructure.getParticle().isElectron()) {
    if(!std::isnan(context.getElSmearingWidth()) ) {
      adaptivePrefactor = context.getElSmearingWidth();
    } else if (!std::isnan(context.getAdaptiveSmearingPrefactor())) {
      adaptivePrefactor = context.getAdaptiveSmearingPrefactor();
    } else {
      adaptivePrefactor = 0.01; // tested to work well for electrons
    }
  }
/*   else if(bandStructure.getParticle().isPhonon()) {
    adaptivePrefactor = 0.05; // this is tested to work well for phonons
  } else {
    adaptivePrefactor = 0.01; // tested to work well for electrons
  } */
  if(mpi->mpiHead())
    std::cout << "The adaptive smearing prefactor is set to " << adaptivePrefactor << std::endl;
}

double AdaptiveGaussianDeltaFunction::getSmearing(const double &energy,
                                           const Eigen::Vector3d &velocity,
                                           [[maybe_unused]] const Eigen::Vector3d &velocity2,
                                           [[maybe_unused]] const Eigen::Vector3d &velocity3) {

  if(velocity2.norm() != 0. || velocity3.norm() != 0.) {
    DeveloperError("Adaptive Gaussian smearing function is being misused.");
  }

  if (velocity.norm() == 0. && energy == 0.) {
    // in this case, velocities are parallel, there shouldn't be
    // scattering unless energy is strictly conserved
    return 1.;
  }

  double sigma = 0.;
  for (int i : {0, 1, 2}) {
    sigma += pow(qTensor.row(i).dot(velocity), 2);
  }
  sigma = prefactor * adaptivePrefactor * sqrt(sigma);

  if (sigma == 0.) {
    return 0.;
  }

  // if the smearing is smaller than 0.1 meV, we renormalize it
  if (sigma < broadeningCutoff ) {
    sigma = broadeningCutoff;
  }

  if (abs(energy) > 2. * sigma)
    return 0.;
  double x = energy / sigma;
  if ( x > 6. ) return 0.; // the error is < 2e-16, and prevents underflow
  // note: the factor ERF_2 corrects for the cutoff at 2*sigma
  return exp(-x * x) / sqrtPi / sigma / erf2;
}

double AdaptiveGaussianDeltaFunction::getSmearing([[maybe_unused]] const double &energy,
                                                  [[maybe_unused]] StateIndex &is) {
  DeveloperError("AdaptiveGaussianDeltaFunction::getSmearing2 not implemented");
  return 1.;
}

// Symmetry respecting adaptive smearing -----------------------------------------

SymAdaptiveGaussianDeltaFunction::SymAdaptiveGaussianDeltaFunction(
    BaseBandStructure &bandStructure, Context& context, double broadeningCutoff_)
    : AdaptiveGaussianDeltaFunction(bandStructure, context, broadeningCutoff_) {

  // cannot override this in the header, as it's an inherited class variable
  // therefore, we have to do it here
  id = DeltaFunction::symAdaptiveGaussian;
}

double SymAdaptiveGaussianDeltaFunction::getSmearing(const double &energy,
                                           const Eigen::Vector3d &velocity,
                                           const Eigen::Vector3d &velocity2,
                                           const Eigen::Vector3d &velocity3) {

  // following section 3.3.1 of the barnaBTE manuscript
  Eigen::Vector3d sigma_ijk = {0.,0.,0.};
  for (int i : {0, 1, 2}) {
    sigma_ijk(0) += pow(qTensor.row(i).dot(velocity), 2);
    sigma_ijk(1) += pow(qTensor.row(i).dot(velocity2), 2);
    sigma_ijk(2) += pow(qTensor.row(i).dot(velocity3), 2);
  }
  double sigma = sigma_ijk.norm();
  sigma = prefactor * sqrt(sigma) * adaptivePrefactor;
  if (sigma == 0.) { return 0.; }

  // NOTE are we sure we want to do we want this cutoff?
  // Seems to affect very little, likely not a big impact either way
  // if the smearing is smaller than 0.0001 eV, we renormalize it
  // for electrons, a common smearing is ~0.01 eV in regular gaussian smearing
  if (sigma < broadeningCutoff ) {
    sigma = broadeningCutoff;
  }

  if (abs(energy) > 2. * sigma)
    return 0.;
  double x = energy / sigma;
  if ( x > 6. ) return 0.; // the error is < 2e-16, and prevents underflow
  // note: the factor ERF_2 corrects for the cutoff at 2*sigma
  return exp(-(x * x)/2.) / (sqrtTwo * sqrtPi) / sigma / erf2;
}

// Tetrahedron method smearing -----------------------------------------

TetrahedronDeltaFunction::TetrahedronDeltaFunction(BaseBandStructure &fullBandStructure_)
    : fullBandStructure(fullBandStructure_),
      fullPoints(fullBandStructure_.getPoints()) {
  auto tup = fullPoints.getMesh();
  auto grid = std::get<0>(tup);
  auto offset = std::get<1>(tup);
  if (offset.norm() > 0.) {
    Error("We didn't implement tetrahedra with offsets");
  }
  if (grid(0) < 1 || grid(1) < 1 || grid(2) < 1) {
    Error("Tetrahedron method initialized with invalid wavevector grid");
  }

  // shifts of 1 k-point in the grid in crystal coordinates
  Eigen::Vector3d deltaGrid;
  for (int i : {0,1,2}) {
    deltaGrid(i) = 1. / grid(i);
  }

  // in this tensor, we store the offset to find the vertices of the
  // tetrahedrons to which the current point belongs to
  // (6: number of tetrahedra, 4: number of vertices, 3: cartesian)
  // multiplying this by the vector Delta k of the grid, we can reconstruct
  // all the points of the tetrahedron.

  subCellShift.resize(8,3);
  subCellShift.row(0) << 0., 0., 0.;
  subCellShift.row(1) << 1., 0., 0.;
  subCellShift.row(2) << 0., 1., 0.;
  subCellShift.row(3) << 1., 1., 0.;
  subCellShift.row(4) << 0., 0., 1.;
  subCellShift.row(5) << 1., 0., 1.;
  subCellShift.row(6) << 0., 1., 1.;
  subCellShift.row(7) << 1., 1., 1.;
  for ( int i=0; i<8; i++) {
    for (int j :  {0,1,2}) {
      subCellShift(i, j) *= deltaGrid(j);
    }
  }

  // list of quadruplets identifying the vertices of the tetrahedra.
  // each number corresponds to the one number corresponds to the (integer) coordinate
  vertices.resize(6,4);
  vertices.row(0) << 0, 1, 2, 5;
  vertices.row(1) << 0, 2, 4, 5;
  vertices.row(2) << 2, 4, 5, 6;
  vertices.row(3) << 2, 5, 6, 7;
  vertices.row(4) << 2, 3, 5, 7;
  vertices.row(5) << 1, 2, 3, 5;
}

double TetrahedronDeltaFunction::getDOS(const double &energy) {
  // initialize tetrahedron weight
  double weight = 0.;
  for (int is : fullBandStructure.irrStateIterator()) {
    auto isIndex = StateIndex(is);
    double degeneracy = double(fullBandStructure.getRotationsStar(isIndex).size());
    weight += getSmearing(energy, isIndex) * degeneracy;
  }
  weight /= double(fullBandStructure.getNumPoints(true));
  return weight;
}

double TetrahedronDeltaFunction::getSmearing(const double &energy,
                                             StateIndex &is) {
  auto t = fullBandStructure.getIndex(is);
  int ik = std::get<0>(t).get();
  auto ibIndex = std::get<1>(t);

  // if the mesh is uniform, each k-point belongs to 6 tetrahedra

  auto kCoordinates = fullPoints.getPointCoordinates(ik, Points::crystalCoordinates);

  // in this tensor, we store the offset to find the vertices of the
  // tetrahedrons to which the current point belongs to
  // (6: number of tetrahedra, 4: number of vertices, 3: cartesian)
  // multiplying this by the vector Delta k of the grid, we can reconstruct
  // all the points of the tetrahedron.

  Eigen::MatrixXd kVectorsSubCell(8,3);
  for ( int i=0; i<8; i++) {
    Eigen::Vector3d x = subCellShift.row(i);
    kVectorsSubCell.row(i) = kCoordinates + x;
  }

  Eigen::VectorXi ikSubCellIndices(8);
  ikSubCellIndices(0) = ik;
  for ( int i=1; i<8; i++) {
    ikSubCellIndices(i) = fullPoints.getIndex(kVectorsSubCell.row(i));
  }

  Eigen::VectorXd energies(8);
  for ( int i=0; i<8; i++) {
    int is1 = fullBandStructure.getIndex(WavevectorIndex(ikSubCellIndices(i)),ibIndex);
    StateIndex is1Idx(is1);
    energies(i) = fullBandStructure.getEnergy(is1Idx);
  }

  // initialize tetrahedron weight
  double numTetra = 0.;
  double weight = 0.;
  for (int iTetra=0; iTetra<6; iTetra++) {
    std::vector<double> tmp(4);
    tmp[0] = energies(vertices(iTetra,0));
    tmp[1] = energies(vertices(iTetra,1));
    tmp[2] = energies(vertices(iTetra,2));
    tmp[3] = energies(vertices(iTetra,3));
    std::sort(tmp.begin(), tmp.end());
    double ee1 = tmp[0];
    double ee2 = tmp[1];
    double ee3 = tmp[2];
    double ee4 = tmp[3];

    // We follow Eq. B6 of Lambin and Vigneron PRB 29 3430 (1984)

    double cnE = 0.;
    if (ee1 <= energy && energy <= ee2) {
      if ( ee2==ee1 || ee3==ee1 || ee4==ee1 ) {
        cnE = 0.;
      } else {
        cnE = 3. * (energy-ee1)*(energy-ee1) / (ee2-ee1) / (ee3-ee1) / (ee4-ee1);
      }
    } else if (ee2 <= energy && energy <= ee3) {

      cnE = 0.;
      if ( ee4==ee2 || ee3==ee2 || ee3==ee1 ) {
        cnE += 0.;
      } else {
        cnE += (ee3-energy)*(energy-ee2) / (ee4-ee2) / (ee3-ee2) / (ee3-ee1);
      }
      if ( ee4==ee1 || ee4==ee2 || ee3==ee1 ) {
        cnE += 0.;
      } else {
        cnE += (ee4-energy)*(energy-ee1) / (ee4-ee1) / (ee4-ee2) / (ee3-ee1);
      }
      cnE *= 3.;

    } else if (ee3 <= energy && energy <= ee4) {
      if ( ee4==ee1 || ee4==ee2 || ee4==ee3 ) {
        cnE = 0.;
      } else {
        cnE = 3. * (ee4-energy)*(ee4-energy) / (ee4-ee1) / (ee4-ee2) / (ee4-ee3);
      }
    }

    // exception
    if ((ee1 == ee2) && (ee1 == ee3) && (ee1 == ee4) & (energy == ee1)) {
      cnE = 0.25;
    }

    numTetra += 1.;

    weight += cnE;
  } // loop over all tetrahedra

  // Zero out extremely small weights
  if (weight < 1.0e-12) {
    weight = 0.;
  }

  // Normalize by number of tetrahedra and the vertices
  weight /= numTetra;
  return weight;
}

double TetrahedronDeltaFunction::getSmearing([[maybe_unused]] const double &energy,
                                             [[maybe_unused]] const Eigen::Vector3d &velocity,
                                             [[maybe_unused]] const Eigen::Vector3d &velocity2,
                                             [[maybe_unused]] const Eigen::Vector3d &velocity3) {

  Error("TetrahedronDeltaFunction getSmearing1 not implemented");
  return 1.;
}
