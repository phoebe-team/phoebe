#ifndef CONTEXT_H
#define CONTEXT_H

#include <limits>  // NaN
#include <string>
#include <vector>

#include "eigen.h"

/** Class containing the user input variables.
 * This class is mostly a container for the input variables.
 *
 * To add a new variable, write a get/set method, and modify setupFromInput().
 */
class Context {
 private:
  std::string phD2FileName = "";
  std::string phD3FileName = "";
  std::string electronH0Name = "";
  std::string wannier90Prefix = "";
  std::string quantumEspressoPrefix = "";
  std::string elPhInterpolation = "";

  std::string appName = "";
  std::string sumRuleD2 = "";
  int smearingMethod = -1;
  double smearingWidth = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd temperatures;
  std::vector<std::string> solverBTE;
  double convergenceThresholdBTE = 1e-5;
  long maxIterationsBTE = 50;

  bool scatteringMatrixInMemory = true;

  std::string windowType = "nothing";
  Eigen::Vector2d windowEnergyLimit;
  double windowPopulationLimit = std::numeric_limits<double>::quiet_NaN();

  Eigen::VectorXd dopings;
  Eigen::VectorXd chemicalPotentials;
  double electronFourierCutoff = std::numeric_limits<double>::quiet_NaN();

  Eigen::Vector3i qMesh;
  Eigen::Vector3i kMesh;

  double fermiLevel = std::numeric_limits<double>::quiet_NaN();
  double numOccupiedStates = std::numeric_limits<double>::quiet_NaN();
  bool hasSpinOrbit = false;

  long dimensionality = 3;

  double dosMinEnergy = std::numeric_limits<double>::quiet_NaN();
  double dosMaxEnergy = std::numeric_limits<double>::quiet_NaN();
  double dosDeltaEnergy = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd inputAtomicPositions;
  Eigen::VectorXi inputAtomicSpecies;
  std::vector<std::string> inputSpeciesNames;

  Eigen::Tensor<double, 3> pathExtrema;
  std::vector<std::string> pathLabels;
  double deltaPath = 0.05;

  double constantRelaxationTime = std::numeric_limits<double>::quiet_NaN();
  bool withIsotopeScattering = true;  // add isotopes in phonon scatt matrix
  Eigen::VectorXd massVariance;       // mass variance for isotope scattering

  // add RTA boundary scattering in phonon scatt matrix
  // boundary length for isotope scattering
  double boundaryLength = std::numeric_limits<double>::quiet_NaN();

  std::string epwFileName = "";
  std::string epaFileName = "";
  double minChemicalPotential = std::numeric_limits<double>::quiet_NaN();
  double maxChemicalPotential = std::numeric_limits<double>::quiet_NaN();
  double deltaChemicalPotential = std::numeric_limits<double>::quiet_NaN();
  double minTemperature = std::numeric_limits<double>::quiet_NaN();
  double maxTemperature = std::numeric_limits<double>::quiet_NaN();
  double deltaTemperature = std::numeric_limits<double>::quiet_NaN();
  double energyRange = std::numeric_limits<double>::quiet_NaN();
  double energyStep = std::numeric_limits<double>::quiet_NaN();
  double eFermiRange = std::numeric_limits<double>::quiet_NaN();

  double epaSmearingEnergy = std::numeric_limits<double>::quiet_NaN();
  double epaDeltaEnergy = std::numeric_limits<double>::quiet_NaN();

  // plot of el-ph coupling
  std::string g2PlotStyle = "";
  Eigen::Vector3d g2PlotFixedPoint;
  std::pair<int,int> g2PlotEl1Bands;
  std::pair<int,int> g2PlotEl2Bands;
  std::pair<int,int> g2PlotPhBands;

  // utilities for parsing

  std::vector<std::string> &split(const std::string &s, char delim,
                                  std::vector<std::string> &elems);
  std::vector<std::string> split(const std::string &s, char delim);

 public:
  // Methods for the apps of plotting the electron-phonon coupling
  std::string getG2PlotStyle();
  void setG2PlotStyle(const std::string x);

  Eigen::Vector3d getG2PlotFixedPoint();
  void setG2PlotFixedPoint(const Eigen::Vector3d x);

  std::pair<int,int> getG2PlotEl1Bands();
  void setG2PlotEl1Bands(const std::pair<int,int> x);

  std::pair<int,int> getG2PlotEl2Bands();
  void setG2PlotEl2Bands(const std::pair<int,int> x);

  std::pair<int,int> getG2PlotPhBands();
  void setG2PlotPhBands(const std::pair<int,int> x);

  //  Setter and getter for all the variables above

  /** gets the name of the file containing the lattice force constants.
   * For Quantum Espresso, this is the path to the file produced by q2r.
   * @return x: the file path.
   */
  std::string getPhD2FileName();
  void setPhD2FileName(const std::string x);

  std::string getPhD3FileName();
  void setPhD3FileName(const std::string x);

  std::string getEpwFileName();
  void setEpwFileName(const std::string x);

  std::string getWannier90Prefix();
  void setWannier90Prefix(const std::string x);
  std::string getQuantumEspressoPrefix();
  void setQuantumEspressoPrefix(const std::string x);
  std::string getElPhInterpolation();

  double getEpaSmearingEnergy();
  double getEpaDeltaEnergy();

  /** gets the name of the file containing the electronic band structure.
   * For Quantum Espresso, this is the path to the XML file.
   * @return path: the file path.
   */
  std::string getElectronH0Name();
  void setElectronH0Name(const std::string x);

  /** gets the value of the cutoff to be used for the Fourier interpolation
   * of the band structure.
   * @return r: the cutoff value.
   */
  double getElectronFourierCutoff();

  /** gets the type of calculation to be run.
   * @return x: the name of the calculation, e.g. "electron-phonon" or
   * "phonon-phonon".
   */
  std::string getAppName();

  /** gets the sum rule to be imposed on the lattice force constants.
   * @return x: the name of the sum rule, i.e. "simple" or "crystal".
   */
  std::string getSumRuleD2();
  void setSumRuleD2(const std::string x);

  /** gets the mesh of points for harmonic phonon properties.
   * @return path: an array with 3 integers representing the q-point mesh.
   */
  Eigen::Vector3i getQMesh();

  /** gets the mesh of points for harmonic electronic properties.
   * @return path: an array with 3 integers representing the k-point mesh.
   */
  Eigen::Vector3i getKMesh();

  /** gets the Window type to be used to filter out states that don't
   * contribute to transport.
   * @return path: an array with 3 integers representing the k-point mesh.
   * @param windowType: a string, which can take values "none", "energy",
   * or "population"
   */
  std::string getWindowType();
  void setWindowType(const std::string x);

  /** gets the values of energy limits to be used with a window on energies.
   * @return x: a vector of 2 doubles representing the minimum and maximum
   * values of energies that will be used
   */
  Eigen::Vector2d getWindowEnergyLimit();

  void setWindowEnergyLimit(const Eigen::Vector2d x);

    /** gets the value of population above which a state is considered active.
     * i.e. the state will be used if its occupation number deviates from 0 or
     * 1 by at least this amount.
     * @return x: the <double> value of the population threshold.
     */
  double getWindowPopulationLimit();
  void setWindowPopulationLimit(const double x);

  /** gets the value of chemical potentials (in Rydbergs) to be used in the
   * calculation of transport properties
   * @return x: the vector of values for chemical potentials
   */
  Eigen::VectorXd getChemicalPotentials();

  /** gets the value of chemical potentials (in Rydbergs) to be used in the
   * calculation of transport properties
   * @return x: the vector of values for chemical potentials
   */
  Eigen::VectorXd getDopings();
  void setDopings(const Eigen::VectorXd x);

  /** gets the value of temperatures (in Rydbergs) to be used in the
   * calculation of transport properties
   * @return x: the vector of values for temperatures
   */
  Eigen::VectorXd getTemperatures();
  void setTemperatures(const Eigen::VectorXd x);

  std::vector<std::string> getSolverBTE();

  double getConvergenceThresholdBTE();

  long getMaxIterationsBTE();

  long getDimensionality();

  double getDosMinEnergy();

  double getDosMaxEnergy();

  double getDosDeltaEnergy();

  // Wannier90 output doesn't contain the crystal information.
  // the user must then supplement it in the input
  // at least, if there's no phonon run
  // we may change the behavior in the future, parsing another file
  Eigen::MatrixXd getInputAtomicPositions();

  Eigen::VectorXi getInputAtomicSpecies();

  std::vector<std::string> getInputSpeciesNames();

  void setInputAtomicPositions(const Eigen::MatrixXd x);
  void setInputAtomicSpecies(const Eigen::VectorXi x);
  void setInputSpeciesNames(const std::vector<std::string> x);

  Eigen::Tensor<double, 3> getPathExtrema();
  std::vector<std::string> getPathLabels();

  double getDeltaPath();

  double getFermiLevel();
  void setFermiLevel(const double &x);

  double getNumOccupiedStates();
  void setNumOccupiedStates(const double &x);

  bool getHasSpinOrbit();
  void setHasSpinOrbit(const bool &x);

  int getSmearingMethod();

  double getSmearingWidth();
  void setSmearingWidth(const double x);

  double getConstantRelaxationTime();

  bool getScatteringMatrixInMemory();
  void setScatteringMatrixInMemory(const bool &x);

  bool getWithIsotopeScattering();

  Eigen::VectorXd getMassVariance();

  double getBoundaryLength();

  // EPA:
  std::string getEpaFileName();
  double getMinChemicalPotential();
  double getMaxChemicalPotential();
  double getDeltaChemicalPotential();
  double getMinTemperature();
  double getMaxTemperature();
  double getDeltaTemperature();
  double getEnergyRange();
  double getEnergyStep();
  double getEFermiRange();

  /** Reads the user-provided input file and saves the input parameters
   * @param fileName: path to the input file
   */
  void setupFromInput(std::string fileName);
};

#endif
