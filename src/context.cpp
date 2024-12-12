#include "context.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <string>
#include <vector>

#include "constants.h"
#include "exceptions.h"
#include "mpiHelper.h"

/** Returns true if a string contains a substring.
 */
bool patternInString(const std::string &s, const std::string &pattern) {
  if (s.find(pattern) != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

/** Parse a string of format "key = value" to return a boolean value.
 */
bool parseBool(std::string &line) {
  std::string delimiter = "=";
  size_t pos = line.find(delimiter);
  std::string s = line.substr(pos + 1);

  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  s.erase(
      std::remove_if(s.begin(), s.end(), [](char c) { return !isalpha(c); }),
      s.end());

  if (s == "false") {
    return false;
  } else if (s == "true") {
    return true;
  } else if (s == "0") {
    return false;
  } else if (s == "1") {
    return true;
  } else {
    Error("Couldn't fix boolean value while parsing");
    return false;
  }
}

/** Parse a string of format "key = value" to return a double value.
 */
double parseDouble(std::string &line) {
  std::string delimiter = "=";
  size_t pos = line.find(delimiter);
  std::string value = line.substr(pos + 1);
  return std::stod(value); // convert to double
}

/** Parse a string of format "key = value units" to return a double value
 * converted in rydberg atomic units.
 */
double parseDoubleWithUnits(std::string &line) {
  double x;

  std::string delimiter = "=";
  size_t pos = line.find(delimiter);
  std::string value = line.substr(pos + 1);
  x = std::stod(value); // convert to double

  // now check the units and convert
  if (patternInString(line, "eV")) {
    x /= energyRyToEv;
  }
  if (patternInString(line, "cmm1")) {
    x /= ryToCmm1;
  }
  if (patternInString(line, "ps")) {
    x /= timeAuToFs * 1.0e-3;
  }
  if (patternInString(line, "fs")) {
    x /= timeAuToFs;
  }
  if (patternInString(line, "mum")) {
    x /= distanceBohrToMum;
  }
  if (patternInString(line, "ang")) {
    x /= distanceBohrToAng;
  }
  if (patternInString(line, "Bohr") || patternInString(line, "bohr")) {
    x /= 1.; // it's already in atomic units
  }

  return x;
}

/** Parse a string of format "key = [value1,value2]" to return a vector double.
 */
std::vector<double> parseDoubleList(std::string &line) {
  std::string delimiter = "[";
  size_t pos1 = line.find_first_of(delimiter);
  delimiter = "]";
  size_t pos2 = line.find_last_of(delimiter);

  if (pos1 == std::string::npos) {
    Error("Error in parseDoubleList");
  }
  if (pos2 == std::string::npos) {
    Error("Error in parseDoubleList");
  }

  std::string s = line.substr(pos1 + 1, pos2 - pos1 - 1);

  std::vector<double> x;
  delimiter = ",";
  while ((pos1 = s.find(delimiter)) != std::string::npos) {
    std::string token = s.substr(0, pos1);
    x.push_back(std::stod(token));
    s.erase(0, pos1 + delimiter.length());
  }
  // Must not forget the last element in the list
  x.push_back(std::stod(s));

  return x;
}

std::tuple<int,double> parseDoubleVectorComponent(std::string line) {
  std::string sep = ")";
  size_t position = line.find(sep);
  std::string part1 = line.substr(0, position);
  part1.erase(std::remove_if(part1.begin(), part1.end(), ::isspace), part1.end());

  std::string sep2 = "=";
  size_t position2 = line.find(sep2);
  std::string part2 = line.substr(position2+1, line.size());
  part2.erase(std::remove_if(part2.begin(), part2.end(), ::isspace), part2.end());

  int idx = std::stoi(part1);
  double val = std::stod(part2);
  return std::make_tuple(idx,val);
}

/** Parse a string of format "key = value units" to return an integer value.
 */
int parseInt(std::string &line) {
  std::string delimiter = "=";
  size_t pos = line.find(delimiter);
  std::string value = line.substr(pos + 1);
  return std::stoi(value); // convert to integer
}

/** Parse a string of format "key = [val1,val2]" to return a vector of ints.
 */
std::vector<int> parseIntList(std::string &line) {
  std::string delimiter = "[";
  size_t pos1 = line.find_first_of(delimiter);
  delimiter = "]";
  size_t pos2 = line.find_last_of(delimiter);

  if (pos1 == std::string::npos) {
    Error("Error in parseIntList");
  }
  if (pos2 == std::string::npos) {
    Error("Error in parseIntList");
  }

  std::string s = line.substr(pos1 + 1, pos2 - pos1 - 1);
  delimiter = ",";
  std::vector<int> x;
  while ((pos1 = s.find(delimiter)) != std::string::npos) {
    std::string token = s.substr(0, pos1);
    x.push_back(std::stoi(token)); // convert to integer
    s.erase(0, pos1 + delimiter.length());
  }
  // Must not forget the last element in the list
  x.push_back(std::stoi(s));

  return x;
}

/** Parse a string of format "key = value" to return a string value.
 */
std::string parseString(std::string &line) {
  std::string delimiter = "'";
  size_t pos1 = line.find_first_of(delimiter);
  size_t pos2 = line.find_last_of(delimiter);

  if (pos1 == std::string::npos) {
    delimiter = "\"";
    pos1 = line.find_first_of(delimiter);
    pos2 = line.find_last_of(delimiter);
    if (pos1 == std::string::npos) {
      Error("Couldn't solve string parsing");
    }
  }

  if (pos1 == pos2) {
    Error("Error parsing string from user input");
  }
  std::string x = line.substr(pos1 + 1, pos2 - pos1 - 1);
  return x;
}

/** Parse a string of format "key = [val1,val2]" to return a vector of strings.
 */
std::vector<std::string> parseStringList(std::string &line) {
  // remove empty spaces
  line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

  std::string delimiter;
  delimiter = "[";
  size_t pos1 = line.find_first_of(delimiter);
  delimiter = "]";
  size_t pos2 = line.find_last_of(delimiter);

  if (pos1 == std::string::npos) {
    Error("Error in parseDoubleList");
  }
  if (pos2 == std::string::npos) {
    Error("Error in parseDoubleList");
  }

  std::string s = line.substr(pos1 + 1, pos2 - pos1 - 1);
  delimiter = ",";
  std::vector<std::string> x;
  while ((pos1 = s.find(delimiter)) != std::string::npos) {
    std::string token = s.substr(0, pos1);
    // we also remove the " symbols
    token.erase(std::remove(token.begin(), token.end(), '"'), token.end());
    x.push_back(token);
    s.erase(0, pos1 + delimiter.length());
  }
  // Must not forget the last element in the list
  s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
  x.push_back(s);

  return x;
}

/** Parse the block of information on the crystal structure.
 * Format:
 * Atom1Name   cartesianCoordX   cartesianCoordY   cartesianCoordZ
 * Atom2Name   cartesianCoordX   cartesianCoordY   cartesianCoordZ
 * ...
 * The coordinates must be provided in Angstroms.
 */
std::tuple<Eigen::MatrixXd, Eigen::VectorXi, std::vector<std::string>>
parseCrystal(std::vector<std::string> &lines) {
  auto numAtoms = int(lines.size());
  Eigen::MatrixXd atomicPositions(numAtoms, 3);
  Eigen::VectorXi atomicSpecies(numAtoms);
  std::vector<std::string> speciesNames;

  int counter = 0;
  for (const std::string &line : lines) {
    // split line by spaces
    std::stringstream ss(line);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    std::vector<std::string> splitLine(begin, end);

    std::string thisElement = splitLine[0];
    // if new element, insert into list of species
    if (std::find(speciesNames.begin(), speciesNames.end(), thisElement) ==
        speciesNames.end()) {
      // thisElement not in speciesNames, add it
      speciesNames.push_back(thisElement);
    }
    // find the index of the current element
    int index = 0;
    for (const auto &speciesName : speciesNames) {
      if (speciesName == thisElement) {
        break;
      }
      index += 1;
    }
    // save species and positions
    atomicSpecies(counter) = index;
    atomicPositions(counter, 0) = std::stod(splitLine[1]);
    atomicPositions(counter, 1) = std::stod(splitLine[2]);
    atomicPositions(counter, 2) = std::stod(splitLine[3]);

    counter++;
  }
  atomicPositions /= distanceBohrToAng;
  return std::make_tuple(atomicPositions, atomicSpecies, speciesNames);
}

/** Reads the block with the information on the path of points in the
 * Brillouin zone.
 *
 * Format:
 * """
 * PointName1  coord1X  coord1Y  coord1Z  PointName2  coord2X  coord2Y  coord2Z
 * PointName2  coord2X  coord2Y  coord2Z  PointName3  coord3X  coord3Y  coord3Z
 * """
 *
 * Each line identifies a segment of the path in the Brillouin zone.
 * PointName1 and PointName2 are the names of the two special points at the
 * opposite ends of the segment.
 * coord1/2 are the crystal coordinates of these special points.
 * The code will pad the segments with equidistant points, as specified by the
 * parameter deltaPath (the distance between points).
 */
std::tuple<std::vector<std::string>, Eigen::Tensor<double, 3>>
parsePathExtrema(std::vector<std::string> &lines) {

  auto numSegments = int(lines.size());
  Eigen::Tensor<double, 3> pathExtrema(numSegments, 2, 3);
  pathExtrema.setZero();
  std::vector<std::string> pathLabels;

  int i = 0;
  for (const std::string &line : lines) {

    // split line by spaces
    std::stringstream ss(line);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    std::vector<std::string> splitLine(begin, end);

    pathLabels.push_back(splitLine[0]);
    pathLabels.push_back(splitLine[4]);

    pathExtrema(i, 0, 0) = std::stod(splitLine[1]);
    pathExtrema(i, 0, 1) = std::stod(splitLine[2]);
    pathExtrema(i, 0, 2) = std::stod(splitLine[3]);

    pathExtrema(i, 1, 0) = std::stod(splitLine[5]);
    pathExtrema(i, 1, 1) = std::stod(splitLine[6]);
    pathExtrema(i, 1, 2) = std::stod(splitLine[7]);

    i++;
  }
  return std::make_tuple(pathLabels, pathExtrema);
}

/** Parse an input block (e.g. for crystal or points path).
 * Returns empty strings/vectors if no block name is found at line "lineCounter"
 * of the input file.
 * Otherwise, returns the block name (i.e. what's named after "begin")
 * and the lines inside the block.
 */
std::tuple<std::string, std::vector<std::string>>
parseBlockNameValue(const std::vector<std::string> &lines,
                    const int &lineCounter) {
  std::string line;
  line = lines[lineCounter];
  if (!patternInString(line, "begin")) {
    std::string empty1;
    std::vector<std::string> empty2;
    return std::make_tuple(empty1, empty2);
  } else {
    std::string pattern = "begin";
    std::size_t found = line.find(pattern);
    std::string blockName = line.substr(found + pattern.size() + 1);
    std::vector<std::string> val;
    for (int unsigned i = lineCounter + 1; i < lines.size(); i++) {
      if (patternInString(lines[i], "end")) {
        break;
      }
      val.push_back(lines[i]);
    }
    return std::make_tuple(blockName, val);
  }
}

std::vector<std::string> &Context::split(const std::string &s, char delimiter,
                                         std::vector<std::string> &elements) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    if (item.length() > 0) {
      elements.push_back(item);
    }
  }
  return elements;
}

/** Split a string by a char delimiter.
 */
std::vector<std::string> Context::split(const std::string &s, char delimiter) {
  std::vector<std::string> elements;
  split(s, delimiter, elements);
  return elements;
}

/** Checks if the line of the input file contains a key=value statement.
 */
bool lineHasParameter(const std::string &line) {
  // parse the line
  if (std::find(line.begin(), line.end(), '=') != line.end()) {
    return true;
  } else {
    return false;
  }
}

/** Given a line of the form "key = value", returns the string "key" and the
 * string containing "value". "value" could contain a string, a list or other
 * things to be further parsed.
 */
std::tuple<std::string, std::string>
parseParameterNameValue(const std::string &line) {
  // we assume that there is "=" in the string
  std::string sep = "=";
  size_t position = line.find(sep); // unsigned integer
  std::string s = line.substr(0, position);
  s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
  std::string val = line.substr(position + 1, line.size());

  // we could also have parenthesis, e.g. mass(43) = 23.;
  std::string sep2 = "(";
  size_t position2 = line.find(sep2); // unsigned integer
  std::string s2 = s.substr(0, position2);
  s2.erase(std::remove_if(s2.begin(), s2.end(), ::isspace), s2.end());
  if (s2 != s) {
    val = line.substr(position2 + 1, line.size());
  }
  return std::make_tuple(s2, val);
}

void Context::setupFromInput(const std::string &fileName) {
  std::vector<std::string> lines;

  // open input file and read content
  {
    std::ifstream infile(fileName);
    std::string line;
    while (std::getline(infile, line)) {
      std::vector<std::string> tokens = split(line, ';');
      for (const std::string &t : tokens) {
        lines.push_back(t);
      }
    }
  }

  int lineCounter = 0;
  for (std::string &line : lines) {

    // remove any part of the line that is beind a # comment first
    line = line.substr(0, line.find_first_of("#"));

    if (line.empty()) { // nothing to do
      lineCounter ++;
      continue;

      // line with pair (key,value)
    } else if (lineHasParameter(line)) {
      auto tup = parseParameterNameValue(line);
      auto parameterName = std::get<0>(tup);
      auto val = std::get<1>(tup);

      if (parameterName == "phFC2FileName") {
        phFC2FileName = parseString(val);
      }
      if (parameterName == "phFC3FileName") {
        phFC3FileName = parseString(val);
      }
      if (parameterName == "phonopyDispFileName") {
        phonopyDispFileName = parseString(val);
      }

      if (parameterName == "phonopyBORNFileName") {
        phonopyBORNFileName = parseString(val);
      }

      if (parameterName == "sumRuleFC2") {
        sumRuleFC2 = parseString(val);
      }
      if (parameterName == "electronH0Name") {
        electronH0Name = parseString(val);
      }
      if (parameterName == "wannier90Prefix") {
        wannier90Prefix = parseString(val);
      }
      if (parameterName == "quantumEspressoPrefix") {
        quantumEspressoPrefix = parseString(val);
      }
      if (parameterName == "elPhInterpolation") {
        elPhInterpolation = parseString(val);
      }
      if (parameterName == "elphFileName") {
        setElphFileName(parseString(val));
      }
      if (parameterName == "electronFourierCutoff") {
        double x = parseDouble(val);
        electronFourierCutoff = x;
      }
      if (parameterName == "qMesh") {
        std::vector<int> vecMesh = parseIntList(val);
        qMesh(0) = vecMesh[0];
        qMesh(1) = vecMesh[1];
        qMesh(2) = vecMesh[2];
      }

      if (parameterName == "kMesh") {
        std::vector<int> vecMesh = parseIntList(val);
        kMesh(0) = vecMesh[0];
        kMesh(1) = vecMesh[1];
        kMesh(2) = vecMesh[2];
      }

      if (parameterName == "windowType") {
        windowType = parseString(val);
      }

      if (parameterName == "windowEnergyLimit") {
        std::vector<double> winLim = parseDoubleList(val);
        windowEnergyLimit[0] = std::min(winLim[0], winLim[1]) / energyRyToEv;
        windowEnergyLimit[1] = std::max(winLim[0], winLim[1]) / energyRyToEv;
      }

      if (parameterName == "windowPopulationLimit") {
        windowPopulationLimit = parseDouble(val);
      }

      if (parameterName == "chemicalPotentials") {
        std::vector<double> x = parseDoubleList(val);
        chemicalPotentials = Eigen::VectorXd::Zero(int(x.size()));
        for (int unsigned i = 0; i < x.size(); i++) {
          chemicalPotentials(i) = x[i] / energyRyToEv;
        }
      }

      if (parameterName == "dopings") {
        std::vector<double> x = parseDoubleList(val);
        dopings = Eigen::VectorXd::Zero(int(x.size()));
        for (int unsigned i = 0; i < x.size(); i++) {
          dopings(i) = x[i];
        }
      }

      if (parameterName == "temperatures") {
        std::vector<double> x = parseDoubleList(val);
        temperatures = Eigen::VectorXd::Zero(int(x.size()));
        for (int unsigned i = 0; i < x.size(); i++) {
          temperatures(i) = x[i] / temperatureAuToSi;
        }
      }

      if (parameterName == "appName") {
        appName = parseString(val);
      }
      if (parameterName == "solverBTE") {
        solverBTE = parseStringList(val);
      }
      if (parameterName == "convergenceThresholdBTE") {
        convergenceThresholdBTE = parseDouble(val);
      }
      if (parameterName == "maxIterationsBTE") {
        maxIterationsBTE = parseInt(val);
      }
      if (parameterName == "dimensionality") {
        dimensionality = parseInt(val);
      }
      if (parameterName == "thickness") {
        thickness = parseDoubleWithUnits(val);
      }
      if (parameterName == "dosMinEnergy") {
        dosMinEnergy = parseDoubleWithUnits(val);
      }
      if (parameterName == "dosMaxEnergy") {
        dosMaxEnergy = parseDoubleWithUnits(val);
      }
      if (parameterName == "dosDeltaEnergy") {
        dosDeltaEnergy = parseDoubleWithUnits(val);
      }
      if (parameterName == "deltaPath") {
        deltaPath = parseDouble(val);
      }
      if (parameterName == "outputEigendisplacements") {
        outputEigendisplacements = parseBool(val);
      }
      if (parameterName == "outputUNTimes") {
        outputUNTimes = parseBool(val);
      }
      if (parameterName == "fermiLevel") {
        fermiLevel = parseDoubleWithUnits(val);
      }
      if (parameterName == "hasSpinOrbit") {
        hasSpinOrbit = parseBool(val);
      }
      if (parameterName == "distributedElPhCoupling") {
        distributedElPhCoupling = parseBool(val);
      }
      if (parameterName == "numOccupiedStates") {
        // note: numOccupiedStates refers to the number of states that are
        // occupied
        // for Wannier: the number of Wannier states that are full
        // for Fourier: the number of occupied bands, minus the states trimmed
        // remember to NOT count the spin degeneracy
        double x = parseDouble(val);
        if (!hasSpinOrbit)
          x *= 2;
        numOccupiedStates = x;
      }
      if (parameterName == "smearingMethod") {
        std::string x_ = parseString(val);
        // TODO: this is hardcoded, should be fixed how we validate input
        if (x_ == "gaussian") {
          smearingMethod = 0;
        } else if (x_ == "adaptiveGaussian") {
          smearingMethod = 1;
        } else if (x_ == "tetrahedron") {
          smearingMethod = 2;
        } else {
          smearingMethod = -1;
        }
      }
      if (parameterName == "smearingWidth") {
        smearingWidth = parseDoubleWithUnits(val);
      }
      if (parameterName == "constantRelaxationTime") {
        constantRelaxationTime = parseDoubleWithUnits(val);
      }
      if (parameterName == "scatteringMatrixInMemory") {
        scatteringMatrixInMemory = parseBool(val);
      }
      if (parameterName == "symmetrizeMatrix") {
        symmetrizeMatrix = parseBool(val);
      }
      if (parameterName == "numRelaxonsEigenvalues") {
        numRelaxonsEigenvalues = parseInt(val);
      }
      if (parameterName == "checkNegativeRelaxons") {
        checkNegativeRelaxons = parseBool(val);
      }
      if (parameterName == "useSymmetries") {
        useSymmetries = parseBool(val);
      }
      if (parameterName == "withIsotopeScattering") {
        withIsotopeScattering = parseBool(val);
      }
      if (parameterName == "masses") {
        std::vector<double> x = parseDoubleList(val);
        customMasses = Eigen::VectorXd::Zero(int(x.size()));
        for (int unsigned i = 0; i < x.size(); i++) {
          customMasses(i) = x[i] * massAmuToRy;
        }
      }
      if (parameterName == "isotopeCouplings") {
        std::vector<double> x = parseDoubleList(val);
        customIsotopeCouplings = Eigen::VectorXd::Zero(int(x.size()));
        for (int unsigned i = 0; i < x.size(); i++) {
          customIsotopeCouplings(i) = x[i];
        }
      }
      if (parameterName == "boundaryLength") {
        boundaryLength = parseDoubleWithUnits(val);
      }

      // EPA
      if (parameterName == "epaFileName") {
        epaFileName = parseString(val);
      }
      if (parameterName == "minChemicalPotential") {
        minChemicalPotential = parseDoubleWithUnits(val);
      }
      if (parameterName == "maxChemicalPotential") {
        maxChemicalPotential = parseDoubleWithUnits(val);
      }
      if (parameterName == "deltaChemicalPotential") {
        deltaChemicalPotential = parseDoubleWithUnits(val);
      }
      if (parameterName == "minTemperature") {
        minTemperature = parseDouble(val) / temperatureAuToSi;
      }
      if (parameterName == "maxTemperature") {
        maxTemperature = parseDouble(val) / temperatureAuToSi;
      }
      if (parameterName == "deltaTemperature") {
        deltaTemperature = parseDouble(val) / temperatureAuToSi;
      }
      if (parameterName == "eFermiRange") {
        eFermiRange = parseDoubleWithUnits(val);
      }
      if (parameterName == "epaSmearingEnergy") {
        epaSmearingEnergy = parseDoubleWithUnits(val);
      }
      if (parameterName == "epaDeltaEnergy") {
        epaDeltaEnergy = parseDoubleWithUnits(val);
      }
      if (parameterName == "epaNumBins") {
        epaNumBins = parseInt(val);
      }
      if (parameterName == "epaMinEnergy") {
        epaMinEnergy = parseDoubleWithUnits(val);
      }
      if (parameterName == "epaMaxEnergy") {
        epaMaxEnergy = parseDoubleWithUnits(val);
      }
      if (parameterName == "epaEnergyRange") {
        epaEnergyRange = parseDoubleWithUnits(val);
      }
      if (parameterName == "epaEnergyStep") {
        epaEnergyStep = parseDoubleWithUnits(val);
      }

      // coupling plot apps

      if (parameterName == "couplingPlotStyle") {
        g2PlotStyle = parseString(val);
      }
      if (parameterName == "couplingMeshStyle") {
        g2MeshStyle = parseString(val);
      }
      if (parameterName == "couplingFixedPoint") {
        std::vector<double> x = parseDoubleList(val);
        for (auto i : {0, 1, 2}) {
          g2PlotFixedPoint(i) = x[i];
        }
      }
      if (parameterName == "couplingPlotBandEl1") {
        std::vector<int> x = parseIntList(val);
        g2PlotEl1Bands.first = x[0];
        g2PlotEl1Bands.second = x[1];
      }
      if (parameterName == "couplingPlotBandEl2") {
        std::vector<int> x = parseIntList(val);
        g2PlotEl2Bands.first = x[0];
        g2PlotEl2Bands.second = x[1];
      }
      if (parameterName == "couplingPlotBandPh") {
        std::vector<int> x = parseIntList(val);
        g2PlotPhBands.first = x[0];
        g2PlotPhBands.second = x[1];
      }
      if (parameterName == "hdf5ElphFileFormat") {
        int x = parseInt(val);
        setHdf5ElPhFileFormat(x);
      }
      if (parameterName == "wsVecFileName") {
        std::string x = parseString(val);
        setWsVecFileName(x);
      }

      // Polarization
      if (parameterName == "numCoreElectrons") {
        std::vector<int> x = parseIntList(val);
        Eigen::VectorXi xx(x.size());
        for (unsigned int i = 0; i < x.size(); i++) {
          xx(i) = x[i];
        }
        setCoreElectrons(xx);
      }

    } else { // it might be a block, or its content

      auto tup = parseBlockNameValue(lines, lineCounter);
      auto blockName = std::get<0>(tup);
      auto value = std::get<1>(tup);

      if (blockName == "crystal") {
        auto tup1 = parseCrystal(value);
        auto inputAtomicPositions_ = std::get<0>(tup1);
        auto inputAtomicSpecies_ = std::get<1>(tup1);
        auto inputSpeciesNames_ = std::get<2>(tup1);
        inputAtomicPositions = inputAtomicPositions_;
        inputAtomicSpecies = inputAtomicSpecies_;
        inputSpeciesNames = inputSpeciesNames_;
      }
      if (blockName == "point path") {
        auto tup2 = parsePathExtrema(value);
        pathLabels = std::get<0>(tup2);
        pathExtrema = std::get<1>(tup2);
      }
    }
    lineCounter += 1;
  }
}

void Context::inputSanityCheck() {

  // disallow symmetries when relaxons are used
  for (const std::string &s : solverBTE) {
    if (s.compare("variational") == 0 || s.compare("relaxons") == 0) {
      if(useSymmetries) { 
        Error("Variational and relaxons solvers cannot be used with symmetries!");
      }
    }
    
    // Warn users about relaxons with even meshes
    if (s.compare("relaxons") == 0) {
      if((qMesh.prod() % 2 == 0 && qMesh.prod() != 0) || (kMesh.prod() % 2 == 0 && kMesh.prod() != 0)) {
        Warning("Relaxons solver should be run with an odd points mesh for reasons of eigenvector parity.");
      }
      if(smearingMethod == 1) { // adaptive smearing
        Warning("Adaptive smearing can cause problems with the quality of the scattering matrix, and may cause negative eigenvalues to appear.\n"
              "If this occurs, you should switch to Gaussian.");
      }
    } 
  }

  // warn the user if thickness is not set but 2d is 
  if(dimensionality == 2) { 
    if(thickness == 1.) { 
      Warning("You have set dimensionality = 2 but not the thickness input parameter.\n"
            "This means your final result will have (height of cell / 1.) applied, when it should be\n"
            "height of cell / thickness!");
    }
  }

}

// helper functions for printInputSummary
template <typename T>
void printVector(const std::string& varName, std::vector<T> vec) {
  std::cout << varName << " = ";
  for (auto i : vec)
    std::cout << " " << i;
  std::cout << std::endl;
}

void printVectorXd(const std::string& varName, Eigen::VectorXd vec,
                   const std::string& unit = "") {
  std::cout << varName << " =";
  for (int i = 0; i < vec.size(); i++)
    std::cout << " " << vec(i);
  std::cout << " (" << unit << ")" << std::endl;
}

// TODO this should be replaced by something in each app like checkRequirements
void Context::printInputSummary(const std::string &fileName) {
  if (!mpi->mpiHead()) return;
  std::cout << std::endl;
  std::cout << "Input read from file: " << fileName << std::endl;
  std::cout << "---------------------------------------------" << std::endl;
  std::cout << std::boolalpha; // make booleans write as true/false

  std::cout << "appName = " << appName << std::endl;

  // crystal structure parameters -------------------
  std::cout << "useSymmetries = " << useSymmetries << std::endl;
  std::cout << "dimensionality = " << dimensionality << std::endl;
  if(dimensionality != 3) std::cout << "thickness = " << thickness * distanceBohrToAng << " ang" << std::endl;
  std::cout << std::endl;

  // phonon parameters -------------------------------
  if (appName.find("honon") != std::string::npos) {
    std::cout << "phFC2FileName = " << phFC2FileName << std::endl;
    std::cout << "sumRuleFC2 = " << sumRuleFC2 << std::endl;
    if (appName == "phononLifetimes" || appName == "phononTransport") {
      std::cout << "phFC3FileName = " << phFC3FileName << std::endl;
    }
    if (!phonopyDispFileName.empty()) {
      std::cout << "phonopyDispFileName = " << phonopyDispFileName << std::endl;
    }
    if (!phonopyBORNFileName.empty()) {
      std::cout << "phonopyBORNFileName = " << phonopyBORNFileName << std::endl;
    }
    std::cout << std::endl;
  }

  // electron and eph parameters
  if (appName.find("elPh") != std::string::npos || appName == "electronLifetimes" ||
        appName == "electronWannierTransport") {

    std::cout << "electronH0Name = " << electronH0Name << std::endl;
    std::cout << "hasSpinOrbit = " << hasSpinOrbit << std::endl;

    if (appName.find("elPh") != std::string::npos || appName == "electronLifetimes" ||
        appName == "electronWannierTransport") {

      std::cout << "elphFileName = " << elphFileName << std::endl;
      if (!elPhInterpolation.empty())
        std::cout << "elPhInterpolation = " << elPhInterpolation << std::endl;
      if (!wannier90Prefix.empty())
        std::cout << "wannier90Prefix = " << wannier90Prefix << std::endl;
      if (!quantumEspressoPrefix.empty())
        std::cout << "quantumEspressoPrefix = " << quantumEspressoPrefix
                  << std::endl;
    }
    // EPA specific parameters
    if (elPhInterpolation == "epa") {
      if (!std::isnan(epaMinEnergy))
        std::cout << "epaMinEnergy = " << epaMinEnergy * energyRyToEv << " eV"
                  << std::endl;
      if (!std::isnan(epaMaxEnergy))
        std::cout << "epaMaxEnergy = " << epaMaxEnergy * energyRyToEv << " eV"
                  << std::endl;
      if (!std::isnan(epaNumBins))
        std::cout << "epaNumBins = " << epaNumBins << std::endl;
      if (!std::isnan(epaSmearingEnergy))
        std::cout << "epaSmearingEnergy = " << epaSmearingEnergy * energyRyToEv
                  << " eV" << std::endl;
      if (!std::isnan(electronFourierCutoff))
        std::cout << "electronFourierCutoff = " << electronFourierCutoff
                  << std::endl;
    }
    if(appName.find("elPhQeToPhoebe") != std::string::npos) {
      std::cout << "distributedElPhCoupling = " << distributedElPhCoupling
        << std::endl;
    }
    // plot coupling apps (either elph or ee)
    if(appName.find("CouplingPlot") != std::string::npos) {
      std::cout << '\n' << "couplingPlotStyle = " << g2PlotStyle << std::endl;
      std::cout << "couplingMeshStyle = " << g2MeshStyle << std::endl;
      if(g2PlotStyle.find("Fixed") != std::string::npos) {
        std::cout << "couplingPlotFixedPoint = [ " << g2PlotFixedPoint.transpose() <<
          " ]" << std::endl;
      }
      if(g2PlotEl1Bands.first != -1)
        std::cout << "couplingPlotEl1Bands = [ " << g2PlotEl1Bands.first << " " <<
                 g2PlotEl1Bands.second << " ]" << std::endl;
      if(g2PlotEl1Bands.first != -1)
        std::cout << "couplingPlotEl2Bands = [ " << g2PlotEl2Bands.first << " " <<
                 g2PlotEl2Bands.second << " ]" << std::endl;
      if(g2PlotEl1Bands.first != -1)
        std::cout << "couplingPlotPhBands = [ " << g2PlotPhBands.first << " " <<
                 g2PlotPhBands.second << " ]" << std::endl;

      if(g2MeshStyle == "pointsPath") {

        std::cout << "deltaPath = " << deltaPath << " 1/Bohr" << std::endl;

        std::cout << "\nBand Path:" << std::endl;
        std::cout << std::setprecision(4) << std::fixed;
        const auto &dim = pathExtrema.dimensions();
        int count = 0;
        for (int i = 0; i < dim[0]; i++) {
          std::cout << pathLabels[count] << " " << pathExtrema(i, 0, 0) << " "
                    << pathExtrema(i, 0, 1) << " " << pathExtrema(i, 0, 2) << "  ";
          count++;
          std::cout << pathLabels[count] << " " << pathExtrema(i, 1, 0) << " "
                    << pathExtrema(i, 1, 1) << " " << pathExtrema(i, 1, 2)
                    << std::endl;
          count++;
        }
      } else {
        if(g2PlotStyle == "qFixed") {
          std::cout << "kMesh = " << kMesh(0) << " " << kMesh(1) << " " << kMesh(2)
                << std::endl;
        }
        else if(g2PlotStyle == "kFixed") {
          std::cout << "qMesh = " << qMesh(0) << " " << qMesh(1) << " " << qMesh(2)
                << std::endl;
        }
      }
    }
    std::cout << std::endl;
  }
  // Qetophoebe doesn't write a nice line after printing
  if (appName.find("elPhQeToPhoebe") != std::string::npos) {
    std::cout << "---------------------------------------------\n" << std::endl;
  }

  // Transport parameters ---------------------------
  if (appName.find("Transport") != std::string::npos ||
      appName.find("Lifetimes") != std::string::npos) {

    if (appName.find("honon") != std::string::npos || appName.find("elPh") != std::string::npos) {
      std::cout << "qMesh = " << qMesh(0) << " " << qMesh(1) << " " << qMesh(2) << std::endl;
      if(!getElphFileName().empty()) {
        std::cout << "electronH0Name = " << electronH0Name << std::endl;
        std::cout << "hasSpinOrbit = " << hasSpinOrbit << std::endl;
        std::cout << "elphFileName = " << elphFileName << std::endl;
      }
    }
    if (appName.find("lectron") != std::string::npos || appName.find("elPh") != std::string::npos
                        || (appName.find("honon") != std::string::npos && !getElphFileName().empty())) {
         std::cout << "kMesh = " << kMesh(0) << " " << kMesh(1) << " " << kMesh(2) << std::endl;
    }

    if (!std::isnan(constantRelaxationTime))
      std::cout << "constantRelaxationTime = " << constantRelaxationTime * timeAuToFs << " fs" << std::endl;

    std::cout << "smearingMethod = ";
    if (smearingMethod == 0)
      std::cout << "gaussian" << std::endl;
    else if (smearingMethod == 1)
      std::cout << "adaptiveGaussian" << std::endl;
    else if (smearingMethod == 2)
      std::cout << "tetrahedron" << std::endl;
    else {
      std::cout << "none" << std::endl;
    }
    if (!std::isnan(smearingWidth))
      std::cout << "smearingWidth = " << smearingWidth * energyRyToEv << " eV"
                << std::endl;

    if(appName.find("Transport") != std::string::npos) {
      std::cout << "solverBTE = RTA";
      for (const auto& i : solverBTE)
        std::cout << ", " << i;
      std::cout << std::endl;
      std::cout << "convergenceThresholdBTE = " << convergenceThresholdBTE << std::endl;
      std::cout << "maxIterationsBTE = " << maxIterationsBTE << std::endl;
    } else {
      std::cout << "solverBTE = RTA" << std::endl;
    }

      std::cout << "scatteringMatrixInMemory = " << scatteringMatrixInMemory
              << std::endl;
      std::cout << "windowType = " << windowType << std::endl;

    if (windowEnergyLimit(0) != 0 || windowEnergyLimit(1) != 0) {
      std::cout << "windowEnergyLimit = " << windowEnergyLimit(0) * energyRyToEv
                << " " << windowEnergyLimit(1) * energyRyToEv << " eV"
                << std::endl;
    }
    if (!std::isnan(windowPopulationLimit)) {
      std::cout << "windowPopulationLimit = " << windowPopulationLimit
                << std::endl;
    }

    printVectorXd("temperatures", temperatures * temperatureAuToSi, "K");
    if (!std::isnan(minTemperature))
      std::cout << "minTemperature = " << minTemperature << "K" << std::endl;
    if (!std::isnan(maxTemperature))
      std::cout << "maxTemperature = " << maxTemperature << "K" << std::endl;
    if (!std::isnan(deltaTemperature))
      std::cout << "deltaTemperature = " << deltaTemperature << "K"
                << std::endl;

    if (appName.find("lectron") != std::string::npos || appName == "phononElectronLifetimes"
                        || (appName.find("honon") != std::string::npos && !getElphFileName().empty())) {
      if (dopings.size() != 0)
        printVectorXd("dopings", dopings, "cm^-3");
      if (chemicalPotentials.size() != 0)
        printVectorXd("chemicalPotentials", chemicalPotentials * energyRyToEv, "eV");
      if (!std::isnan(minChemicalPotential))
        std::cout << "minChemicalPotential = "
                  << minChemicalPotential * energyRyToEv << " eV" << std::endl;
      if (!std::isnan(maxChemicalPotential))
        std::cout << "maxChemicalPotential = "
                  << maxChemicalPotential * energyRyToEv << " eV" << std::endl;
      if (!std::isnan(deltaChemicalPotential))
        std::cout << "deltaChemicalPotential = "
                  << deltaChemicalPotential * energyRyToEv << " eV" << std::endl;
      if (!std::isnan(eFermiRange))
        std::cout << "eFermiRange = " << eFermiRange << " eV" << std::endl;
      if (!std::isnan(fermiLevel))
        std::cout << "fermiLevel = " << fermiLevel * energyRyToEv << std::endl;
      if (!std::isnan(numOccupiedStates)) {
        if(!hasSpinOrbit) { // need to account for spin factor
          std::cout << "numOccupiedStates = " << numOccupiedStates/2.0 << std::endl;
        } else {
          std::cout << "numOccupiedStates = " << numOccupiedStates << std::endl;
        }
      }
    }
    // should not be printed when phellifetimes app is run
    if (appName.find("honon") != std::string::npos && appName.find("lectron") == std::string::npos) {
      std::cout << "withIsotopeScattering = " << withIsotopeScattering << std::endl;
      if (customMasses.size() != 0)
        printVectorXd("masses", customMasses*massRyToAmu, "amu");
      if (customIsotopeCouplings.size() != 0)
        std::cout << "isotopeCouplings = " << customIsotopeCouplings.transpose() << "\n";
      if (!std::isnan(boundaryLength))
        std::cout << "boundaryLength = " << boundaryLength * distanceBohrToMum << " mum" << std::endl;
      std::cout << "outputUNTimes = " << outputUNTimes << std::endl;
    }
    std::cout << "---------------------------------------------\n" << std::endl;
  }

  // dos variables ---------------------------------------
  if (appName.find("Dos") != std::string::npos) {
    std::cout << "dosMinEnergy = " << dosMinEnergy * energyRyToEv << " eV"
              << std::endl;
    std::cout << "dosMaxEnergy = " << dosMaxEnergy * energyRyToEv << " eV"
              << std::endl;
    std::cout << "dosDeltaEnergy = " << dosDeltaEnergy * energyRyToEv << " eV"
              << std::endl;
    if (appName.find("Fourier") != std::string::npos) {
      std::cout << "electronFourierCutoff = " << electronFourierCutoff
                << std::endl;
    }
    std::cout << "---------------------------------------------\n" << std::endl;
  }

  // band structure variables ----------------------------
  if (appName.find("Bands") != std::string::npos) {
    const auto &dim = pathExtrema.dimensions();
    std::cout << "deltaPath = " << deltaPath << " 1/Bohr" << std::endl;
    if (appName.find("Fourier") != std::string::npos) {
      std::cout << "electronFourierCutoff = " << electronFourierCutoff
                << std::endl;
    }
    if (appName.find("phonon") != std::string::npos) {
      std::cout << "outputEigendisplacements = " << outputEigendisplacements <<
        std::endl;
    }

    std::cout << "\nBand Path:" << std::endl;
    std::cout << std::setprecision(4) << std::fixed;
    int count = 0;
    for (int i = 0; i < dim[0]; i++) {
      std::cout << pathLabels[count] << " " << pathExtrema(i, 0, 0) << " "
                << pathExtrema(i, 0, 1) << " " << pathExtrema(i, 0, 2) << "  ";
      count++;
      std::cout << pathLabels[count] << " " << pathExtrema(i, 1, 0) << " "
                << pathExtrema(i, 1, 1) << " " << pathExtrema(i, 1, 2)
                << std::endl;
      count++;
    }
    std::cout << "---------------------------------------------\n" << std::endl;
  }

  // epa variables  ----------------------------------------
  if (appName == "transportEpa") {
    std::cout << "epaFileName = " << epaFileName << std::endl;
    std::cout << "electronH0Name = " << electronH0Name << std::endl;
    std::cout << "hasSpinOrbit = " << hasSpinOrbit << std::endl;
    std::cout << "kMesh = " << kMesh(0) << " " << kMesh(1) << " " << kMesh(2)
              << std::endl;
    if (!std::isnan(epaEnergyRange))
      std::cout << "epaEnergyRange = " << epaEnergyRange * energyRyToEv << " eV"
                << std::endl;
    if (!std::isnan(epaEnergyStep))
      std::cout << "epaEnergyStep = " << epaEnergyStep * energyRyToEv << " eV"
                << std::endl;
    if (!std::isnan(epaMinEnergy))
      std::cout << "epaMinEnergy = " << epaMinEnergy * energyRyToEv << " eV"
                << std::endl;
    if (!std::isnan(epaMaxEnergy))
      std::cout << "epaMaxEnergy = " << epaMaxEnergy * energyRyToEv << " eV"
                << std::endl;
    if (!std::isnan(epaSmearingEnergy))
      std::cout << "epaSmearingEnergy = " << epaSmearingEnergy * energyRyToEv
                << " eV" << std::endl;
    if (!std::isnan(epaDeltaEnergy))
      std::cout << "epaDeltaEnergy = " << epaDeltaEnergy * energyRyToEv << " eV"
                << std::endl;
    if (!std::isnan(electronFourierCutoff))
      std::cout << "electronFourierCutoff = " << electronFourierCutoff
                << std::endl;

    printVectorXd("temperatures", temperatures * temperatureAuToSi, "K");
    if (!std::isnan(minTemperature))
      std::cout << "minTemperature = " << minTemperature << "K" << std::endl;
    if (!std::isnan(maxTemperature))
      std::cout << "maxTemperature = " << maxTemperature << "K" << std::endl;
    if (!std::isnan(deltaTemperature))
      std::cout << "deltaTemperature = " << deltaTemperature << "K"
                << std::endl;
    if (dopings.size() != 0)
      printVectorXd("dopings", dopings, "cm^-3");
    if (chemicalPotentials.size() != 0)
      printVectorXd("chemicalPotentials", chemicalPotentials * energyRyToEv,
                    "eV");
    if (!std::isnan(minChemicalPotential))
      std::cout << "minChemicalPotential = "
                << minChemicalPotential * energyRyToEv << " eV" << std::endl;
    if (!std::isnan(maxChemicalPotential))
      std::cout << "maxChemicalPotential = "
                << maxChemicalPotential * energyRyToEv << " eV" << std::endl;
    if (!std::isnan(deltaChemicalPotential))
      std::cout << "deltaChemicalPotential = "
                << deltaChemicalPotential * energyRyToEv << " eV" << std::endl;
    if (!std::isnan(eFermiRange))
      std::cout << "eFermiRange = " << eFermiRange << " eV" << std::endl;
    if (!std::isnan(fermiLevel))
      std::cout << "fermiLevel = " << fermiLevel * energyRyToEv << std::endl;
    if (!std::isnan(numOccupiedStates))
      std::cout << "numOccupiedStates = " << numOccupiedStates << std::endl;
    std::cout << "---------------------------------------------\n" << std::endl;
  }

  inputSanityCheck(); // call the input sanity check to make sure these make sense

}

std::string Context::getPhFC2FileName() { return phFC2FileName; }
void Context::setPhFC2FileName(const std::string &x) { phFC2FileName = x; }

std::string Context::getPhFC3FileName() { return phFC3FileName; }
void Context::setPhFC3FileName(const std::string &x) { phFC3FileName = x; }

std::string Context::getPhonopyDispFileName() { return phonopyDispFileName; }
/* just used as a test function */
void Context::setPhonopyDispFileName(const std::string &x) {
  phonopyDispFileName = x;
}
std::string Context::getPhonopyBORNFileName() { return phonopyBORNFileName; }

std::string Context::getSumRuleFC2() { return sumRuleFC2; }
void Context::setSumRuleFC2(const std::string &x) { sumRuleFC2 = x; }

std::string Context::getElphFileName() { return elphFileName; }
void Context::setElphFileName(const std::string &x) { elphFileName = x; }

std::string Context::getElectronH0Name() { return electronH0Name; }
void Context::setElectronH0Name(const std::string &x) { electronH0Name = x; }

std::string Context::getWannier90Prefix() { return wannier90Prefix; }
void Context::setWannier90Prefix(const std::string &x) { wannier90Prefix = x; }

std::string Context::getQuantumEspressoPrefix() {
  return quantumEspressoPrefix;
}
void Context::setQuantumEspressoPrefix(const std::string &x) {
  quantumEspressoPrefix = x;
}

std::string Context::getElPhInterpolation() { return elPhInterpolation; }

double Context::getEpaSmearingEnergy() const { return epaSmearingEnergy; }
double Context::getEpaDeltaEnergy() const { return epaDeltaEnergy; }
double Context::getEpaMinEnergy() const { return epaMinEnergy; }
double Context::getEpaMaxEnergy() const { return epaMaxEnergy; }
int Context::getEpaNumBins() const { return epaNumBins; }
double Context::getEpaEnergyRange() const { return epaEnergyRange; }
double Context::getEpaEnergyStep() const { return epaEnergyStep; }

double Context::getElectronFourierCutoff() const {
  return electronFourierCutoff;
}

std::string Context::getAppName() { return appName; }

Eigen::Vector3i Context::getQMesh() { return qMesh; }

Eigen::Vector3i Context::getKMesh() { return kMesh; }

std::string Context::getWindowType() { return windowType; }
void Context::setWindowType(const std::string &x) { windowType = x; }

Eigen::Vector2d Context::getWindowEnergyLimit() { return windowEnergyLimit; }

void Context::setWindowEnergyLimit(const Eigen::Vector2d &x) {
  windowEnergyLimit = x;
}

double Context::getWindowPopulationLimit() const {
  return windowPopulationLimit;
}
void Context::setWindowPopulationLimit(const double &x) {
  windowPopulationLimit = x;
}

Eigen::VectorXd Context::getChemicalPotentials() { return chemicalPotentials; }

Eigen::VectorXd Context::getDopings() { return dopings; }

void Context::setDopings(const Eigen::VectorXd &x) { dopings = x; }

Eigen::VectorXd Context::getTemperatures() { return temperatures; }

void Context::setTemperatures(const Eigen::VectorXd &x) { temperatures = x; }

std::vector<std::string> Context::getSolverBTE() { return solverBTE; }

double Context::getConvergenceThresholdBTE() const {
  return convergenceThresholdBTE;
}

int Context::getMaxIterationsBTE() const { return maxIterationsBTE; }

int Context::getDimensionality() const { return dimensionality; }

double Context::getThickness() const { return thickness; }

double Context::getDosMinEnergy() const { return dosMinEnergy; }

double Context::getDosMaxEnergy() const { return dosMaxEnergy; }

double Context::getDosDeltaEnergy() const { return dosDeltaEnergy; }

Eigen::MatrixXd Context::getInputAtomicPositions() {
  return inputAtomicPositions;
}

Eigen::VectorXi Context::getInputAtomicSpecies() { return inputAtomicSpecies; }

std::vector<std::string> Context::getInputSpeciesNames() {
  return inputSpeciesNames;
}

void Context::setInputAtomicPositions(const Eigen::MatrixXd &x) {
  inputAtomicPositions = x;
}
void Context::setInputAtomicSpecies(const Eigen::VectorXi &x) {
  inputAtomicSpecies = x;
}
void Context::setInputSpeciesNames(const std::vector<std::string> &x) {
  inputSpeciesNames = x;
}

Eigen::Tensor<double, 3> Context::getPathExtrema() { return pathExtrema; }
std::vector<std::string> Context::getPathLabels() { return pathLabels; }

double Context::getDeltaPath() const { return deltaPath; }

bool Context::getOutputEigendisplacements() const { return outputEigendisplacements; }
bool Context::getOutputUNTimes() const { return outputUNTimes; }

double Context::getFermiLevel() const { return fermiLevel; }

void Context::setFermiLevel(const double &x) { fermiLevel = x; }

double Context::getNumOccupiedStates() const { return numOccupiedStates; }

void Context::setNumOccupiedStates(const double &x) { numOccupiedStates = x; }

bool Context::getHasSpinOrbit() const { return hasSpinOrbit; }

void Context::setHasSpinOrbit(const bool &x) { hasSpinOrbit = x; }

int Context::getSmearingMethod() const { return smearingMethod; }

double Context::getSmearingWidth() const { return smearingWidth; }
void Context::setSmearingWidth(const double &x) { smearingWidth = x; }

double Context::getConstantRelaxationTime() const {
  return constantRelaxationTime;
}

bool Context::getScatteringMatrixInMemory() const {
  return scatteringMatrixInMemory;
}
void Context::setScatteringMatrixInMemory(const bool &x) {
  scatteringMatrixInMemory = x;
}

bool Context::getSymmetrizeMatrix() const {
  return symmetrizeMatrix;
}
void Context::setSymmetrizeMatrix(const bool &x) {
  symmetrizeMatrix = x;
}

int Context::getNumRelaxonsEigenvalues() const {
  return numRelaxonsEigenvalues;
}
void Context::setNumRelaxonsEigenvalues(const int &x) {
  numRelaxonsEigenvalues = x;
}

bool Context::getCheckNegativeRelaxons() const { return checkNegativeRelaxons; }

bool Context::getUseSymmetries() const { return useSymmetries; }
void Context::setUseSymmetries(const bool &x) { useSymmetries = x; }

Eigen::VectorXd Context::getMasses() { return customMasses; }
Eigen::VectorXd Context::getIsotopeCouplings() { return customIsotopeCouplings; }

bool Context::getWithIsotopeScattering() const { return withIsotopeScattering; }

double Context::getBoundaryLength() const { return boundaryLength; }

std::string Context::getEpaFileName() { return epaFileName; }

double Context::getMinChemicalPotential() const { return minChemicalPotential; }

double Context::getMaxChemicalPotential() const { return maxChemicalPotential; }

double Context::getDeltaChemicalPotential() const {
  return deltaChemicalPotential;
}

double Context::getMinTemperature() const { return minTemperature; }

double Context::getMaxTemperature() const { return maxTemperature; }

double Context::getDeltaTemperature() const { return deltaTemperature; }

double Context::getEFermiRange() const { return eFermiRange; }

std::string Context::getG2PlotStyle() { return g2PlotStyle; }
void Context::setG2PlotStyle(const std::string &x) { g2PlotStyle = x; }

std::string Context::getG2MeshStyle() { return g2MeshStyle; }
void Context::setG2MeshStyle(const std::string &x) { g2MeshStyle = x; }

Eigen::Vector3d Context::getG2PlotFixedPoint() { return g2PlotFixedPoint; }
void Context::setG2PlotFixedPoint(const Eigen::Vector3d &x) {
  g2PlotFixedPoint = x;
}

std::pair<int, int> Context::getG2PlotEl1Bands() { return g2PlotEl1Bands; }
void Context::setG2PlotEl1Bands(const std::pair<int, int> &x) {
  g2PlotEl1Bands = x;
}

std::pair<int, int> Context::getG2PlotEl2Bands() { return g2PlotEl2Bands; }
void Context::setG2PlotEl2Bands(const std::pair<int, int> &x) {
  g2PlotEl2Bands = x;
}

std::pair<int, int> Context::getG2PlotPhBands() { return g2PlotPhBands; }
void Context::setG2PlotPhBands(const std::pair<int, int> &x) {
  g2PlotPhBands = x;
}

Eigen::VectorXi Context::getCoreElectrons() { return numCoreElectrons; }

void Context::setCoreElectrons(const Eigen::VectorXi &x) {
  for (unsigned int i = 0; i < x.size(); i++) {
    if (x(i) < 0) {
      Error("Found negative number of core electrons");
    }
  }
  numCoreElectrons = x;
}

bool Context::getDistributedElPhCoupling() const {
  return distributedElPhCoupling;
}

void Context::setDistributedElPhCoupling(const bool &x) {
  distributedElPhCoupling = x;
}

int Context::getHdf5ElPhFileFormat() const {
  return hdf5ElphFileFormat;
}

void Context::setHdf5ElPhFileFormat(const int &x) {
  hdf5ElphFileFormat = x;
}

std::string Context::getWsVecFileName() const {
  return wsVecFileName;
}

void Context::setWsVecFileName(const std::string& x) {
  wsVecFileName = x;
}
