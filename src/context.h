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
  std::string phFC2FileName;
  std::string phFC3FileName;
  std::string phonopyDispFileName;
  std::string phonopyBORNFileName;

  std::string electronH0Name;
  std::string wannier90Prefix;
  std::string jdftxScfOutFile;
  std::string quantumEspressoPrefix;
  std::string elPhInterpolation;

  std::string appName;
  std::string sumRuleFC2;
  int smearingMethod = -1;
  double smearingWidth = std::numeric_limits<double>::quiet_NaN();
  double elSmearingWidth = std::numeric_limits<double>::quiet_NaN();
  double phSmearingWidth = std::numeric_limits<double>::quiet_NaN();
  //double dragSmearingWidth = std::numeric_limits<double>::quiet_NaN();
  double adaptiveSmearingPrefactor = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd temperatures;
  std::vector<std::string> solverBTE;
  double convergenceThresholdBTE = 1e-2;
  int maxIterationsBTE = 50;

  bool scatteringMatrixInMemory = true;
  bool useSymmetries = false;
  bool symmetrizeBandStructure = false;

  std::string windowType = "nothing";
  Eigen::Vector2d windowEnergyLimit = Eigen::Vector2d::Zero();
  double windowPopulationLimit = std::numeric_limits<double>::quiet_NaN();

  Eigen::VectorXd dopings;
  Eigen::VectorXd chemicalPotentials;
  double electronFourierCutoff = std::numeric_limits<double>::quiet_NaN();

  Eigen::Vector3i qMesh = Eigen::Vector3i::Zero();
  Eigen::Vector3i kMesh = Eigen::Vector3i::Zero();
  Eigen::Vector3i kMeshPhEl = Eigen::Vector3i::Zero();

  double fermiLevel = std::numeric_limits<double>::quiet_NaN();
  double numOccupiedStates = std::numeric_limits<double>::quiet_NaN();
  bool hasSpinOrbit = false;

  int dimensionality = 3;

  double dosMinEnergy = std::numeric_limits<double>::quiet_NaN();
  double dosMaxEnergy = std::numeric_limits<double>::quiet_NaN();
  double dosDeltaEnergy = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd inputAtomicPositions;
  Eigen::VectorXi inputAtomicSpecies;
  std::vector<std::string> inputSpeciesNames;

  Eigen::Tensor<double, 3> pathExtrema;
  std::vector<std::string> pathLabels;
  double deltaPath = 0.05;

  bool outputEigendisplacements = false; // used by bands app if phonon eigdisps are dumped
  bool outputUNTimes = false; // triggers scattering matrix to output times for U and N processes

  double constantRelaxationTime = std::numeric_limits<double>::quiet_NaN();
  bool withIsotopeScattering = true;  // add isotopes in phonon scattering matrix

  // for custom masses and custom isotope scattering
  Eigen::VectorXd customIsotopeCouplings;
  Eigen::VectorXd customMasses;

  // add RTA boundary scattering in phonon scattering matrix
  // boundary length for isotope scattering
  double boundaryLength = std::numeric_limits<double>::quiet_NaN();

  std::string elphFileName;
  double minChemicalPotential = std::numeric_limits<double>::quiet_NaN();
  double maxChemicalPotential = std::numeric_limits<double>::quiet_NaN();
  double deltaChemicalPotential = std::numeric_limits<double>::quiet_NaN();
  double minTemperature = std::numeric_limits<double>::quiet_NaN();
  double maxTemperature = std::numeric_limits<double>::quiet_NaN();
  double deltaTemperature = std::numeric_limits<double>::quiet_NaN();
  double eFermiRange = std::numeric_limits<double>::quiet_NaN();

  std::string epaFileName;
  double epaEnergyRange = std::numeric_limits<double>::quiet_NaN();
  double epaEnergyStep = std::numeric_limits<double>::quiet_NaN();
  double epaMinEnergy = std::numeric_limits<double>::quiet_NaN();
  double epaMaxEnergy = std::numeric_limits<double>::quiet_NaN();
  int epaNumBins;
  double epaSmearingEnergy = std::numeric_limits<double>::quiet_NaN();
  double epaDeltaEnergy = std::numeric_limits<double>::quiet_NaN();

  // plot of el-ph coupling
  std::string g2PlotStyle = "allToAll";
  std::string g2MeshStyle = "pointsMesh";
  Eigen::Vector3d g2PlotFixedPoint = {0,0,0};
  std::pair<int,int> g2PlotEl1Bands = std::make_pair(0,-1);
  std::pair<int,int> g2PlotEl2Bands = std::make_pair(0,-1);
  std::pair<int,int> g2PlotPhBands = std::make_pair(0,-1);

  // utilities for parsing

  static std::vector<std::string> &split(const std::string &s, char delimiter,
                                  std::vector<std::string> &elements);
  static std::vector<std::string> split(const std::string &s, char delimiter);

  // variable used for the polarization
  Eigen::VectorXi numCoreElectrons;

  bool distributedElPhCoupling = true; // MPI parallelize the el-ph coupling
  // currently only support parallelization of the qe2Phoebe app

  // if true, enforce the symmetrization of the scattering matrix
  bool symmetrizeMatrix = false;
  // if true, and symmetrize matrix = false,
  // only compute the upper triangle of the matrix
  bool useUpperTriangle = false;

  // number of eigenvalues to use the in the relaxons solver
  int numRelaxonsEigenvalues = 0;
  // toggle the check for negative relaxons eigenvalues in few eigenvalues case
  bool checkNegativeRelaxons = true;
  // toggle the enforcement of the matrix being positive semidefinite
  bool enforcePositiveSemiDefinite = false; 

  // for coupled transport
  bool useDragTerms = true;

  int hdf5ElphFileFormat = 1;
  std::string wsVecFileName;

public:

  /** Reads the user-provided input file and saves the input parameters
   * @param fileName: path to the input file
   */
  void setupFromInput(const std::string &fileName);

  /** Prints the user-provided input variables to output, including default values.
   * @param fileName: path to the input file, just to print where input came from.
   */
  void printInputSummary(const std::string &fileName);

  /** Reinforce that all variables dependent on each other are correct
   */
  void checkDependentVariables(); 

  // Setter and getter for all the variables above ----------------

  // Methods for the apps of plotting the electron-phonon coupling
  std::string getG2PlotStyle();
  void setG2PlotStyle(const std::string &x);

  std::string getG2MeshStyle();
  void setG2MeshStyle(const std::string &x);

  Eigen::Vector3d getG2PlotFixedPoint();
  void setG2PlotFixedPoint(const Eigen::Vector3d &x);

  std::pair<int,int> getG2PlotEl1Bands();
  void setG2PlotEl1Bands(const std::pair<int,int> &x);

  std::pair<int,int> getG2PlotEl2Bands();
  void setG2PlotEl2Bands(const std::pair<int,int> &x);

  std::pair<int,int> getG2PlotPhBands();
  void setG2PlotPhBands(const std::pair<int,int> &x);

  /** gets the name of the file containing the lattice force constants.
   * For Quantum Espresso, this is the path to the file produced by q2r.
   * @return x: the file path.
   */
  std::string getPhFC2FileName();
  void setPhFC2FileName(const std::string &x);

  std::string getPhFC3FileName();
  void setPhFC3FileName(const std::string &x);

  std::string getPhonopyDispFileName();
  void setPhonopyDispFileName(const std::string &x);

  std::string getPhonopyBORNFileName();

  std::string getElphFileName();
  void setElphFileName(const std::string &x);

  std::string getWannier90Prefix();
  void setWannier90Prefix(const std::string &x);

  std::string getQuantumEspressoPrefix();
  void setQuantumEspressoPrefix(const std::string &x);

  std::string getJDFTxScfOutFile();
  void setJDFTxScfOutFile(const std::string &x);

  std::string getElPhInterpolation();

  double getEpaSmearingEnergy() const;
  double getEpaDeltaEnergy() const;
  double getEpaMinEnergy() const;
  double getEpaMaxEnergy() const;
  int getEpaNumBins() const;

  /** gets the name of the file containing the electronic band structure.
   * For Quantum Espresso, this is the path to the XML file.
   * @return path: the file path.
   */
  std::string getElectronH0Name();
  void setElectronH0Name(const std::string &x);

  /** gets the value of the cutoff to be used for the Fourier interpolation
   * of the band structure.
   * @return r: the cutoff value.
   */
  double getElectronFourierCutoff() const;

  /** gets the type of calculation to be run.
   * @return x: the name of the calculation, e.g. "electron-phonon" or
   * "phonon-phonon".
   */
  std::string getAppName();

  /** gets the sum rule to be imposed on the lattice force constants.
   * @return x: the name of the sum rule, i.e. "simple" or "crystal".
   */
  std::string getSumRuleFC2();
  void setSumRuleFC2(const std::string &x);

  /** gets the mesh of points for harmonic phonon properties.
   * @return path: an array with 3 integers representing the q-point mesh.
   */
  Eigen::Vector3i getQMesh();

  /** gets the mesh of points for harmonic electronic properties.
   * @return path: an array with 3 integers representing the k-point mesh.
   */
  Eigen::Vector3i getKMesh();

  /** getter for the kMesh used to sample the fermi surface
  * in phEl scattering calculation
  * @return mesh: the Kmesh used in phEl scattering calculation */
  Eigen::Vector3i getKMeshPhEl();
  /** setter for the kMesh used to sample the fermi surface
  * in phEl scattering calculation
  * @param kmesh: the Kmesh used in phEl scattering calculation */
  void setKMeshPhEl(Eigen::Vector3i newKMeshPhEl);

  /** gets the Window type to be used to filter out states that don't
   * contribute to transport.
   * @return path: an array with 3 integers representing the k-point mesh.
   * @param windowType: a string, which can take values "none", "energy",
   * or "population"
   */
  std::string getWindowType();
  void setWindowType(const std::string &x);

  /** gets the values of energy limits to be used with a window on energies.
   * @return x: a vector of 2 doubles representing the minimum and maximum
   * values of energies that will be used
   */
  Eigen::Vector2d getWindowEnergyLimit();
  void setWindowEnergyLimit(const Eigen::Vector2d &x);

  /** gets the value of population above which a state is considered active.
   * i.e. the state will be used if its occupation number deviates from 0 or
   * 1 by at least this amount.
   * @return x: the <double> value of the population threshold.
   */
  double getWindowPopulationLimit() const;
  void setWindowPopulationLimit(const double &x);

  /** gets the value of chemical potentials (in Rydberg) to be used in the
   * calculation of transport properties
   * @return x: the vector of values for chemical potentials
   */
  Eigen::VectorXd getChemicalPotentials();

  /** gets the value of chemical potentials (in Rydberg) to be used in the
   * calculation of transport properties
   * @return x: the vector of values for chemical potentials
   */
  Eigen::VectorXd getDopings();
  void setDopings(const Eigen::VectorXd &x);

  /** gets the value of temperatures (in Rydberg) to be used in the
   * calculation of transport properties
   * @return x: the vector of values for temperatures
   */
  Eigen::VectorXd getTemperatures();
  void setTemperatures(const Eigen::VectorXd &x);

  std::vector<std::string> getSolverBTE();

  double getConvergenceThresholdBTE() const;

  int getMaxIterationsBTE() const;

  int getDimensionality() const;

  double getDosMinEnergy() const;

  double getDosMaxEnergy() const;

  double getDosDeltaEnergy() const;

  // Wannier90 output doesn't contain the crystal information.
  // the user must then supplement it in the input
  // at least, if there's no phonon run
  // we may change the behavior in the future, parsing another file
  Eigen::MatrixXd getInputAtomicPositions();

  Eigen::VectorXi getInputAtomicSpecies();

  std::vector<std::string> getInputSpeciesNames();

  void setInputAtomicPositions(const Eigen::MatrixXd &x);
  void setInputAtomicSpecies(const Eigen::VectorXi &x);
  void setInputSpeciesNames(const std::vector<std::string> &x);

  Eigen::Tensor<double, 3> getPathExtrema();
  std::vector<std::string> getPathLabels();

  double getDeltaPath() const;

  bool getOutputEigendisplacements() const;
  bool getOutputUNTimes() const;

  double getFermiLevel() const;
  void setFermiLevel(const double &x);

  double getNumOccupiedStates() const;
  void setNumOccupiedStates(const double &x);

  bool getHasSpinOrbit() const;
  void setHasSpinOrbit(const bool &x);

  int getSmearingMethod() const;

  double getSmearingWidth() const;
  void setSmearingWidth(const double &x);

  double getElSmearingWidth() const;
  void setElSmearingWidth(const double &x);

  double getPhSmearingWidth() const;
  void setPhSmearingWidth(const double &x);

  //double getDragSmearingWidth() const;
  //void setDragSmearingWidth(const double &x);

  double getAdaptiveSmearingPrefactor() const;
  void setAdaptiveSmearingPrefactor(const double &x);

  double getConstantRelaxationTime() const;

  bool getScatteringMatrixInMemory() const;
  void setScatteringMatrixInMemory(const bool &x);

  bool getUseSymmetries() const;
  void setUseSymmetries(const bool &x);
  bool getSymmetrizeBandStructure() const;
  void setSymmetrizeBandStructure(const bool &x);

  bool getWithIsotopeScattering() const;

  Eigen::VectorXd getMasses();
  Eigen::VectorXd getIsotopeCouplings();

  double getBoundaryLength() const;

  // EPA:
  std::string getEpaFileName();
  double getMinChemicalPotential() const;
  double getMaxChemicalPotential() const;
  double getDeltaChemicalPotential() const;
  double getMinTemperature() const;
  double getMaxTemperature() const;
  double getDeltaTemperature() const;
  double getEpaEnergyRange() const;
  double getEpaEnergyStep() const;
  double getEFermiRange() const;

  Eigen::VectorXi getCoreElectrons();
  void setCoreElectrons(const Eigen::VectorXi &x);

  bool getDistributedElPhCoupling() const;
  void setDistributedElPhCoupling(const bool &x);

  int getHdf5ElPhFileFormat() const;
  void setHdf5ElPhFileFormat(const int &x);

  std::string getWsVecFileName() const;
  void setWsVecFileName(const std::string& x);

  bool getUseDragTerms() const;
  void setUseDragTerms(const bool &x);

  // relaxons variables

  bool getSymmetrizeMatrix() const;
  void setSymmetrizeMatrix(const bool &x);

  bool getUseUpperTriangle() const;
  void setUseUpperTriangle(const bool &x);

  int getNumRelaxonsEigenvalues() const;
  void setNumRelaxonsEigenvalues(const int &x);

  bool getCheckNegativeRelaxons() const;
  bool setCheckNegativeRelaxons() const;

  bool getEnforcePositiveSemiDefinite() const;
  void setEnforcePositiveSemiDefinite(const bool &x);

};

#endif
