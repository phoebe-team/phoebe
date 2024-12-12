#ifndef SCATTERING_H
#define SCATTERING_H

#include "Matrix.h"
#include "context.h"
#include "delta_function.h"
#include "vector_bte.h"

/** Base class of the scattering matrix.
 * Note: this is an abstract class, which can only work if builder() is defined
 */
class ScatteringMatrix {
public:
  /** Scattering matrix constructor
   * @param context: object containing user input configuration.
   * @param statisticsSweep: object controlling the loops over temperature
   * and chemical potential.
   * @param innerBandStructure: bandStructure object. This is the mesh used
   * to integrate the properties for each state of outerBandStructure.
   * For transport calculation, this object is typically equal to
   * outerBandStructure. Might differ when outerBS is on a path of points.
   * @param outerBandStructure: bandStructure object. The anharmonic
   * properties are computed on the grid of points specified by this object,
   * e.g. this could be a path of points or a uniform mesh of points
   */
  ScatteringMatrix(Context &context_, StatisticsSweep &statisticsSweep_,
                   BaseBandStructure &innerBandStructure_,
                   BaseBandStructure &outerBandStructure_);

  /** Destructor
   */
  ~ScatteringMatrix();

  /** This method needs to be called right after the constructor.
   * It will setup variables of the scattering matrix, and compute at least
   * the linewidths.
   * If the user input ask to store the matrix in memory, the scattering
   * matrix gets allocated and built here, through a call to a virtual member
   * builder(), which is defined in subclasses.
   */
  void setup();

  /** Returns the diagonal matrix elements.
   *  For the case of electrons, this is the linewidths -- A_out = Linewidths
   *  For the case of phonons, this is A_out = linewidths * n(n+1)
   * @return diagonal: a VectorBTE containing the diagonal of the scattering matrix.
   */
  VectorBTE diagonal();

  /** Get and set operator.
   * Returns the stored value if the matrix element (row,col) is stored in
   * memory by the MPI process, otherwise returns zero.
   * Just calls this on the underlying PMatrix object
   */
  double& operator()(const int &row, const int &col);

  /** Computes the product A*f - diagonal(A)*f
   * where A is the scattering matrix and f is the vector of quasiparticle
   * populations.
   */
  VectorBTE offDiagonalDot(VectorBTE &inPopulation);
  std::vector<VectorBTE> offDiagonalDot(std::vector<VectorBTE> &inPopulations);

  /** Computes the product A*f
   * where A is the scattering matrix and f is the vector of quasiparticle
   * populations.
   */
  VectorBTE dot(VectorBTE &inPopulation);
  std::vector<VectorBTE> dot(std::vector<VectorBTE> &inPopulations);

//  /** Computes the product A*B, where A is the scattering matrix, and
//   * B is an Eigen::MatrixXd. This can be used to compute products of the
//   * scattering matrix with other vectors.
//   */
//  ParallelMatrix<double> dot(const ParallelMatrix<double> &otherMatrix);

  /** Call to obtain the single-particle relaxation times of the systems
   * @param int: which times to output, 0 = internalDiagonal, 1 = normal, 2 = umklapp
   * @return tau: a VectorBTE object storing the relaxation times
   */
  VectorBTE getSingleModeTimes(const int whichTimes = 0);

  /** Call to obtain the single-particle linewidths.
   *
   * Note that the definition of linewidths is slightly different between
   * electrons and phonons. For phonons, linewidths satisfy the relation
   * Gamma * Tau = hBar (in atomic units, Gamma * Tau = 1). This Tau is the
   * same Tau that enters the Boltzmann equation, and Gamma/Tau are related to
   * the scattering operator as
   * A_{ii} = n_i(n_i+1) / tau_i = n_i(n_i+1) * Gamma_i
   *
   * For electrons, we are using a modified definition
   * A_{ii} = f_i(1-f_i) / tau_i = Gamma_i
   * where Gamma_i is the imaginary part of the self-energy, and Tau is the
   * transport relaxation time that enters the transport equations, and satisfy
   * the relation Gamma_i * Tau_i = f_i * (1-f_i)
   *
   * Regardless, this function just returns Gamma
   *
   * @return linewidths: a VectorBTE object storing the linewidths
   */
  VectorBTE getLinewidths();

  /** Call to set the single-particle linewidths.
   *
   *  See getLinewidths function for notes about linewidths.
   *  @param linewidths: this is Gamma, without any population factors
   */
  void setLinewidths(VectorBTE &linewidths);

  /** Converts the scattering matrix from the form A to the symmetrised Omega.
   * A acts on the canonical phonon population f, while Omega acts on the
   * symmetrised phonon population \tilde{n}.
   * Note: for phonons, n = bose(bose+1)f , with bose being the
   * bose--einstein distribution, while n = sqrt(bose(bose+1)) tilde(n).
   * Only works if the matrix is kept in memory and after setup() has been
   * called.
   */
  void a2Omega();

//  /** The inverse of a2Omega, converts the matrix Omega to A
//   */
//  void omega2A();

  /** Diagonalize the scattering matrix
   * @param numEigenvalues: if a number is supplied, calculate
   *                only the first few of these points
   * @return eigenvalues: a Eigen::VectorXd with the eigenvalues
   * @return eigenvectors: a Eigen::MatrixXd with the eigenvectors
   * Eigenvectors are aligned on rows: eigenvectors(qpState,eigenIndex)
   */
  std::tuple<Eigen::VectorXd, ParallelMatrix<double>>
                                diagonalize(int numEigenvalues = 0);

  /** Outputs the quantity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToJSON(const std::string &outFileName);

  /** Function to combine a BTE index and a cartesian index into one index of
   * the scattering matrix. If no symmetries are used, the output is equal to
   * the BteIndex.
   *
   * @param bteIndex: BteIndex for a Bloch state entering the BTE
   * @param cartIndex: CartIndex for a cartesian direction
   * @return sMatrixIndex: and index identifying the scattering matrix row/col.
   *
   * Note: when symmetries are used, the scattering matrix has indices (i,j)
   * where i,j = (BteIndex,CartIndex) combined together, where CartIndex is an
   * index on cartesian directions, and BteIndex an index on the Bloch states
   * entering the Boltzmann equation.
   */
  int getSMatrixIndex(BteIndex &bteIndex, CartIndex &cartIndex);

  /** Function to split a scattering matrix index (on rows/columns of S) into
   * a BTE index and a cartesian index. If no symmetries are used, the output
   * is equal to the BteIndex and CartIndex is set to 0. It is suggested to skip
   * this function if symmetries are NOT used.
   *
   * @param bteIndex: BteIndex for a Bloch state entering the BTE
   * @param cartIndex: CartIndex for a cartesian direction
   * @return sMatrixIndex: and index identifying the scattering matrix row/col.
   *
   * Note: when symmetries are used, the scattering matrix has indices (i,j)
   * where i,j = (BteIndex,CartIndex) combined together, where CartIndex is an
   * index on cartesian directions, and BteIndex an index on the Bloch states
   * entering the Boltzmann equation.
   */
  std::tuple<BteIndex, CartIndex> getSMatrixIndex(const int &iMat);

  /** Reinforce the condition that the matrix is symmetric
   */
  void symmetrize();

  void relaxonsToJSON(const std::string& fileName, const Eigen::VectorXd& eigenvalues);

  /** Average the coupling for degenerate states.
   * When there are degenerate energies, the freedom of choice of eigenvectors
   * within the corresponding eigenspace should not affect the final coupling.
   * We therefore average the coupling over the degenerate states.
   * @param coupling: Calculated coupling to be averaged.
   * @param energies1: Energies to check for degeneracy in ib1.
   * @param energies2: Energies to check for degeneracy in ib2.
   * @param energies3: Energies to check for degeneracy in ib3.
   */
  static void symmetrizeCoupling(Eigen::Tensor<double,3>& coupling,
                        const Eigen::VectorXd& energies1,
                        const Eigen::VectorXd& energies2,
                        const Eigen::VectorXd& energies3);

  /** Call the underlying PMatrix function to return the iterator of all elements of the
   * matrix which are local
   * @return: an iterator of local state index pairs
   **/
  std::vector<std::tuple<int, int>> getAllLocalStates();

 protected:
  Context &context;
  StatisticsSweep &statisticsSweep;

  // Smearing is a pointer created in constructor with a smearing factory
  // Used in the construction of the scattering matrix to approximate the
  // Dirac-delta function in transition rates.
  DeltaFunction *smearing;

  BaseBandStructure &innerBandStructure;
  BaseBandStructure &outerBandStructure;

  // constant relaxation time approximation -> the matrix is just a scalar
  // and there are simplified evaluations taking place
  bool constantRTA = false;
  bool highMemory = true;     // whether the matrix is kept in memory
  bool outputUNTimes = false;    // whether to output U and N processes in RTA

  // NOTE: about the definition of A
  // A acts on the canonical population, omega on the population
  // A acts on f, Omega on n, with n = bose(bose+1)f e.g. for phonons
  //
  // Generally: A_out = diagonal of the scattering matrix.
  // In the case of phonons, A_out = linewidth * n(n+1)
  // In the case of electrons, A_out = linewidht
  bool isMatrixOmega = false; // whether the matrix is Omega or A


  // we save the diagonal matrix element in a dedicated vector
  VectorBTE internalDiagonal;
  std::shared_ptr<VectorBTE> internalDiagonalUmklapp;
  std::shared_ptr<VectorBTE> internalDiagonalNormal;
  // the scattering matrix, initialized if highMemory==true
  ParallelMatrix<double> theMatrix;

  int numStates; // number of Bloch states (i.e. the size of theMatrix)
  int numPoints; // number of wavevectors
  int numCalculations;  // number of "Calculations", i.e. number of temperatures and
  // chemical potentials on which we compute scattering.
  int dimensionality_;

  // this is to exclude problematic Bloch states, e.g. acoustic phonon modes
  // at gamma, which have zero frequencies and thus have several non-analytic
  // behaviors (e.g. Bose--Einstein population=infinity). We set to zero
  // terms related to these states.
  std::vector<int> excludeIndices;

  /** Method that actually computes the scattering matrix.
   * Pure virtual function: needs an implementation in every subclass.
   * Builder has three behaviors:
   * 1) if matrix.size()==0 and linewidth is passed, builder computes the
   * quasiparticle linewidths.
   * 2) if matrix.size > 0 and linewidth is passed, builder computes the
   * quasiparticle linewidths and the scattering matrix. Memory intensive!
   * 3) if matrix.size()==0, linewidth is not passed, but we pass in+out
   * populations, we compute outPopulation = scatteringMatrix * inPopulation.
   * This doesn't require to store the matrix in memory.
   */
  virtual void builder(VectorBTE *linewidth,
                       std::vector<VectorBTE> &inPopulations,
                       std::vector<VectorBTE> &outPopulations) = 0;

  /** Returns a vector of pairs of wavevector indices to iterate over during
   * the construction of the scattering matrix.
   * @param switchCase: if 0, returns the pairs of wavevectors to loop for
   * the case where the scattering matrix is built in memory.
   * If != 0, returns the pairs of wavevectors to loop for the case where
   * only the action of the scattering matrix is computed.
   * @param rowMajor: set to true if the loop is in the form
   * for iq1 { for iq2 {...}}. False for the opposite (default).
   * @return vector<tuple<iq1,iq2>>: a tuple of wavevector indices to loop
   * over in the construction of the scattering matrix.
   *
   * Implementation: note that ph and el scattering are implemented differently.
   * In detail, ph is computed as (for iq2) {(for iq1)}.
   * instead el is computed as (for ik1) {(for ik2)}.
   * where index1 is computed over irreducible points and iq2 over reducible
   * points. Therefore, we have to distinguish the two cases (they're inverted).
   * Also, we distinguish the case of matrix in memory or not.
   * For scattering matrix in memory, we parallelize over the matrix elements
   * owned by an MPI process. Otherwise, we trivially parallelize over the outer
   * loop on points.
   */
  std::vector<std::tuple<std::vector<int>, int>>
  getIteratorWavevectorPairs(const int &switchCase,
                             const bool &rowMajor = false);

  /** Performs an average of the linewidths over degenerate states.
   * This is necessary since the coupling |V_3| is not averaged over different
   * states, and hence introduces differences between degenerate states.
   * Note: don't use this function when computing the scattering matrix, but
   * only for computing lifetimes or the RTA approximation. If one wants to do
   * something similar when computing the full scattering matrix (in memory)
   * one should either average the coupling or somehow average rows and columns
   * of the scattering matrix.
   * @param linewidth: a pointer (the one passed to builder()) containing the
   * linewidths to be averaged.
   */
  void degeneracyAveragingLinewidths(VectorBTE *linewidth);

  /** Internal helper to formats single mode times stored in vectorBTE
   * object based on if the matrix isOmega or not.
   * @param VectorBTE& diagonal: the list of times we want to reformat
   * @return VectorBTE: containing the reformated times
   */
  VectorBTE getTimesFromVectorBTE(VectorBTE& diagonal);

};

#endif
