#include "bands_app.h"
#include "constants.h"
#include "eigen.h"
#include "mpiHelper.h"
#include "points.h"
#include "parser.h"
#include "qe_input_parser.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

// forward declare this helper function, so we can leave the important
// run functions at the top
void outputBandsToJSON(FullBandStructure &fullBandStructure, Context &context,
                       Points &pathPoints, std::string outFileName);

/* --------------------- PhononBandsApp ------------------------------ */
void PhononBandsApp::run(Context &context) {

  if (mpi->mpiHead()) {
    std::cout << "Starting phonon bands calculation" << std::endl;
  }

  // Read the necessary input files
  auto tup = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(tup);
  auto phononH0 = std::get<1>(tup);

  // first we compute the band structure on the fine grid
  Points pathPoints(crystal, context.getPathExtrema(), context.getDeltaPath());
  bool withVelocities = false;
  bool withEigenvectors = context.getOutputEigendisplacements();
  FullBandStructure fullBandStructure =
      phononH0.populate(pathPoints, withVelocities, withEigenvectors);

  // arguments: bandStructure, context, pathPoints, outputFileName
  outputBandsToJSON(fullBandStructure, context, pathPoints,
                    "phonon_bands.json");

  if (mpi->mpiHead()) {
    std::cout << "Finishing phonon bands calculation" << std::endl;
  }
}

/* --------------------- ElectronWannierBandsApp --------------------- */
void ElectronWannierBandsApp::run(Context &context) {
  if (mpi->mpiHead()) {
    std::cout << "Starting electron (Wannier) bands calculation" << std::endl;
  }

  // Read the necessary input files
  auto tup = Parser::parseElHarmonicWannier(context);
  auto crystal = std::get<0>(tup);
  auto electronH0 = std::get<1>(tup);

  // first we compute the band structure on the fine grid
  Points pathPoints(crystal, context.getPathExtrema(), context.getDeltaPath());

  bool withVelocities = false;
  bool withEigenvectors = false;
  FullBandStructure fullBandStructure =
      electronH0.populate(pathPoints, withVelocities, withEigenvectors);

  // arguments: bandStructure, context, pathPoints, outputFileName
  outputBandsToJSON(fullBandStructure, context, pathPoints,
                    "electron_bands.json");

  if (mpi->mpiHead()) {
    std::cout << "Finishing electron (Wannier) bands calculation" << std::endl;
  }
}

/* --------------------- ElectronFourierBandsApp --------------------- */
void ElectronFourierBandsApp::run(Context &context) {
  if (mpi->mpiHead()) {
    std::cout << "Starting electron (Fourier) bands calculation" << std::endl;
  }

  // Read the necessary input files
  auto tup = Parser::parseElHarmonicFourier(context);
  auto crystal = std::get<0>(tup);
  auto electronH0 = std::get<1>(tup);

  // first we compute the band structure on the fine grid
  Points pathPoints(crystal, context.getPathExtrema(), context.getDeltaPath());

  bool withVelocities = false;
  bool withEigenvectors = false;
  FullBandStructure fullBandStructure =
      electronH0.populate(pathPoints, withVelocities, withEigenvectors);

  // arguments: bandStructure, context, pathPoints, outputFileName
  outputBandsToJSON(fullBandStructure, context, pathPoints,
                    "electron_bands.json");

  if (mpi->mpiHead()) {
    std::cout << "Finishing electron (Fourier) bands calculation" << std::endl;
  }
}

//https://github.com/nlohmann/json/issues/1510
// for writing a complex number to json
namespace std {
  template <class T> void to_json(nlohmann::json &j, const std::complex< T > &p) {
    j = nlohmann::json {p.real(), p.imag()};
  }
}

/* helper function to output bands to a json file */
void outputBandsToJSON(FullBandStructure &fullBandStructure, Context &context,
                       Points &pathPoints, std::string outFileName) {

  if (!mpi->mpiHead())
    return;

  std::vector<std::vector<double>> outEnergies;
  std::vector<int> wavevectorIndices;
  std::vector<double> tempEns;
  std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> eigendisplacements;
  std::vector<std::vector<double>> vecCrystal;
  std::vector<std::vector<double>> vecAtomPos;
  std::vector<std::vector<double>> pathCoordinates;
  auto particle = fullBandStructure.getParticle();
  // mev for phonons
  double energyConversion = particle.isPhonon() ? 1000*energyRyToEv : energyRyToEv;
  int numBands = fullBandStructure.getNumBands();

  // determine path extrema to output to json
  std::vector<std::vector<double>> extremaCoordinates;
  std::vector<int> pathLabelIndices;
  Eigen::Tensor<double, 3> pathExtrema = context.getPathExtrema();
  auto numExtrema = pathExtrema.dimensions();
  for (int pe = 0; pe < numExtrema[0]; pe++) {
    // store coordinates of the extrema
    extremaCoordinates.push_back(
        {pathExtrema(pe, 0, 0), pathExtrema(pe, 0, 1), pathExtrema(pe, 0, 2)});
    extremaCoordinates.push_back(
        {pathExtrema(pe, 1, 0), pathExtrema(pe, 1, 1), pathExtrema(pe, 1, 2)});
  }

  // store the wavevector indices, wavevectors and high sym point indices
  unsigned int extremaCount =
      0; // keeps track of how many high sym points we've found
  for (int ik = 0; ik < pathPoints.getNumPoints(); ik++) {

    // store wavevector indices
    wavevectorIndices.push_back(ik);
    auto ikIndex = WavevectorIndex(ik);

    // store the path coordinates
    auto coord = pathPoints.getPointCoordinates(ik);
    pathCoordinates.push_back({coord[0], coord[1], coord[2]});
    auto pathLabels = context.getPathLabels();
    // check if this point is one of the high sym points,
    // and if it is, save the index
    if (coord[0] == extremaCoordinates[extremaCount][0] &&
        coord[1] == extremaCoordinates[extremaCount][1] &&
        coord[2] == extremaCoordinates[extremaCount][2]) {

      pathLabelIndices.push_back(ik);
      // if the next extrema is the same as this one, add the index
      //  twice and increment extrema count twice.
      // we have to do this because the point won't occur a second time
      // in the path list.
      // Alternatively, if this pathLabel and the next pathLabel are the same,
      // even if the extremaCoordinates are not the same, they are considered the same point.
      // This can happen when two band path points are sym equiv, such as the X points
      // 0.5 0 0.5 and 0.5 -0.5 0 in the FCC lattice.
      if (extremaCount + 1 < extremaCoordinates.size()) { // check bounds
        if (extremaCoordinates[extremaCount] == extremaCoordinates[extremaCount + 1]
                || pathLabels[extremaCount] == pathLabels[extremaCount + 1]) {
          pathLabelIndices.push_back(ik);
          extremaCount += 2;
        } else {
          extremaCount++;
        }
      }
    }

    // store the energies
    Eigen::VectorXd energies = fullBandStructure.getEnergies(ikIndex);
    for (int ib = 0; ib < numBands; ib++) {
      tempEns.push_back(energies(ib) * energyConversion);
    }
    outEnergies.push_back(tempEns);
    tempEns.clear();

    if(context.getOutputEigendisplacements() && particle.isPhonon()) {
      // these should be "with mass scaling" which would make them phonon
      // eigendisplacements already
      WavevectorIndex kIdx = WavevectorIndex(ik);
      Eigen::Tensor<std::complex<double>, 3> eigendisps = fullBandStructure.getPhEigenvectors(kIdx);
      std::vector<std::vector<std::vector<std::complex<double>>>> tempEigendisps;
      for (int ib = 0; ib < numBands; ib++) {
        std::vector<std::vector<std::complex<double>>> atomDisps;
        for (int iat = 0; iat < numBands/3; iat++) {
          std::vector<std::complex<double>> atomDisp;
          for (int iDim = 0; iDim < 3; iDim++) {
            std::complex<double> angDisp = eigendisps(iDim,iat,ib) * distanceBohrToAng;
            atomDisp.push_back(angDisp);
          }
          atomDisps.push_back(atomDisp);
        }
        tempEigendisps.push_back(atomDisps);
      }
      eigendisplacements.push_back(tempEigendisps);
    }
  }

  // if we're outputting eigenvectors, we also likely want crystal and atom pos
  if(context.getOutputEigendisplacements() && particle.isPhonon()) {
    Crystal crystal = fullBandStructure.getPoints().getCrystal();
    auto atomPositions = crystal.getAtomicPositions();

    for (int ialpha = 0; ialpha < 3; ialpha++) {
      std::vector<double> lvec;
      for (int ibeta = 0; ibeta < 3; ibeta++) {
         lvec.push_back(crystal.getDirectUnitCell()(ialpha,ibeta) * distanceBohrToAng);
      }
      vecCrystal.push_back(lvec);
    }
    for (int iat = 0; iat < numBands/3; iat++) {
      std::vector<double> atomPos;
      for (int iDim = 0; iDim < 3; iDim++) {
        atomPos.push_back(atomPositions(iat,iDim) * distanceBohrToAng);
      }
      vecAtomPos.push_back(atomPos);
    }
  }

  // output to json
  nlohmann::json output;
  output["wavevectorIndices"] = wavevectorIndices;
  output["wavevectorCoordinates"] = pathCoordinates;
  output["highSymLabels"] = context.getPathLabels();
  output["highSymIndices"] = pathLabelIndices;
  output["highSymCoordinates"] = extremaCoordinates;
  output["numBands"] = numBands;
  output["energies"] = outEnergies;
  output["particleType"] = particle.isPhonon() ? "phonon" : "electron";
  output["energyUnit"] = particle.isPhonon() ? "meV" : "eV";
  output["coordsType"] = "lattice";
  if(context.getOutputEigendisplacements() && particle.isPhonon()) {
    output["phononEigendisplacements"] = eigendisplacements;
    output["coordsTypePhononEigendisplacements"] = "cartesian";
    output["latticeVectors"] = vecCrystal;
    output["distanceUnit"] = "Angstrom";
    output["atomPositionsCartesian"] = vecAtomPos;
    output["atomSpecies"] = fullBandStructure.getPoints().getCrystal().getAtomicNames();
  }
  // if the user supplied mu, we will output that as well
  // if not, we don't include mu
  //if (!std::isnan(context.getFermiLevel()) && particle.isElectron()) {
  //  output["fermiLevel"] = context.getFermiLevel() * energyConversion;
  //}
  std::ofstream o(outFileName);
  o << std::setw(3) << output << std::endl;
  o.close();
}

void PhononBandsApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getPhFC2FileName(), "PhFC2FileName");
  throwErrorIfUnset(context.getPathExtrema(), "points path extrema");
  throwErrorIfUnset(context.getDeltaPath(), "deltaPath");
  throwWarningIfUnset(context.getSumRuleFC2(), "sumRuleFC2");
}

void ElectronWannierBandsApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getPathExtrema(), "points path extrema");
  throwErrorIfUnset(context.getDeltaPath(), "deltaPath");

  std::string crystalMsg = "crystal structure";
  throwErrorIfUnset(context.getInputAtomicPositions(), crystalMsg);
  throwErrorIfUnset(context.getInputSpeciesNames(), crystalMsg);
  throwErrorIfUnset(context.getInputAtomicSpecies(), crystalMsg);
}

void ElectronFourierBandsApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getPathExtrema(), "points path extrema");
  throwErrorIfUnset(context.getDeltaPath(), "deltaPath");
  throwErrorIfUnset(context.getElectronFourierCutoff(),
                    "electronFourierCutoff");
}
