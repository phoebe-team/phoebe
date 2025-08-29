#ifndef SCATTERING_H
#define SCATTERING_H

#include "Matrix.h"
#include "context.h"
#include "delta_function.h"
#include "vector_bte.h"

// 3 cases:
// we compute and store in memory the scattering matrix and the diagonal
// we compute the action of the scattering matrix on the in vector, returning outVec = sMatrix*vector
// we compute only the linewidths
enum MatrixCase {
  fullMatrix,
  matrixVectorProduct,
  linewidthOnly
};

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
   * to integrate the scattering rates for each state of outerBandStructure.
   * For transport calculation, this object is typically equal to
   * outerBandStructure. Might differ when outerBS is on a path of points.
   * @param outerBandStructure: bandStructure object. The scattering rates
   * properties are computed on the grid of points specified by this object,
   * e.g. this could be a path of points or a uniform mesh of points
   */
  ScatteringMatrix(Context &context_, StatisticsSweep &statisticsSweep_,
                   BaseBandStructure &innerBandStructure_,
                   BaseBandStructure &outerBandStructure_);

  /** Default constructor */
  ScatteringMatrix();

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
   * @return tau: a VectorBTE object storing the relaxation times
   */
  VectorBTE getSingleModeTimes(); 
  
  /** Converts symmetrized internal scattering matrix diagonal to 
  * linewidths and returns the result.
  * See general function of the same name for full notes
  * @return linewidths: a VectorBTE object storing the linewidths
  */
  VectorBTE getLinewidths();
 
  /** Call to set the single-particle linewidths.
   *
   *  See getLinewidths function for notes about linewidths.
   *  @param linewidths: this is Gamma, without any population factors
   */
  void setLinewidths(VectorBTE &linewidths);
  
  /** Call to obtain the single-particle relaxation times of the systems
   * given any internal diagonal like object -- (basically, the linewidths but 
   * potentially scaled by some symmetrization factor)
   * @param internalDiagonal: a VectorBTE object storing the linewidths (potentially 
   * 	symmetrization scaled)
   * @return tau: a VectorBTE object storing the relaxation times
   */
   VectorBTE getSingleModeTimes(const VectorBTE& anyInternalDiagonal);

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
    * @param anyInteralDiagonal : vector bte or pointer to vector BTE
    * @return linewidths: a VectorBTE object storing the linewidths
    */
   VectorBTE getLinewidths(const VectorBTE& anyInternalDiagonal);

  /** Converts the scattering matrix from the form A to the symmetrised Omega.
   * A acts on the canonical phonon population f, while Omega acts on the
   * symmetrised phonon population \tilde{n}.
   * Note: for phonons, n = bose(bose+1)f , with bose being the
   * bose--einstein distribution, while n = sqrt(bose(bose+1)) tilde(n).
   * Only works if the matrix is kept in memory and after setup() has been
   * called.
   */
  void a2Omega();

  /** The inverse of a2Omega, converts the matrix Omega to A
   */
  void omega2A();

  /** Diagonalize the scattering matrix
   * @param numEigenvalues: if a number is supplied, calculate
   *                only the first few of these points
   * @return eigenvalues: a Eigen::VectorXd with the eigenvalues
   * @return eigenvectors: a Eigen::MatrixXd with the eigenvectors
   * Eigenvectors are aligned on rows: eigenvectors(qpState,eigenIndex)
   */
  std::tuple<Eigen::VectorXd, ParallelMatrix<double>>
                                diagonalize(int numEigenvalues = 0);

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

  /** Output relaxons scattering matrix quantities to file (particularly the tau values)
  * Jenny's note: However, I have a feeling this should be elsewhere, as we're actually passing
  * the eigenvalues back into this function.
  * @param fileName: the name of the file to write out to
  * @param eigenvalues: the tau values from a relaxons solve
  **/
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

  // IO operations ---------

  /** Outputs the matrix to an hdf5 file.
   * @param outFileName: string representing the name of the json file
   */
  void outputToHDF5(const std::string &outFileName);

 protected:

  Context &context;
  StatisticsSweep &statisticsSweep;

  // Smearing is a pointer created in constructor with a smearing factory
  // Used in the construction of the scattering matrix to approximate the
  // Dirac-delta function in transition rates.
  //DeltaFunction *smearing; // moved to specific ph and el base scatteringMatrix objs

  BaseBandStructure &innerBandStructure;
  BaseBandStructure &outerBandStructure;
  
  enum MatrixCase matrixCase;
  
  // constant relaxation time approximation -> the matrix is just a scalar
  // and there are simplified evaluations taking place
  bool constantRTA = false;
  bool highMemory = true;     // whether the matrix is kept in memory
  bool outputUNTimes = false;    // whether to output U and N processes in RTA

  double boundaryLength;
  bool doBoundary;

  // NOTE: about the definition of A
  // A acts on the canonical population, omega on the population
  // A acts on f, Omega on n, with n = bose(bose+1)f e.g. for phonons
  //
  // Generally: A_out = diagonal of the scattering matrix.
  // In the case of phonons, A_out = linewidth * n(n+1)
  // In the case of electrons, A_out = linewidht
  bool isMatrixOmega = false; // whether the matrix is Omega or A

  // if we're using the coupled matrix, we need to use this to
  // save the scattering rates differently
  bool isCoupled = false;

  /* function which shifts the indices to the correct quadrant in the CBTE case. 
  * for a pure ph or el matrix, does nothing (default defined here)
  * @param iBte1, iBte2: the bte indices used to index this quadrant.
  * @param p1, p2: the particle types indicating the desired quadrant
  * @return: a tuple containing the shifted indices
  */
  std::function<std::tuple<long,long>(long, long, const Particle&, const Particle &)> shiftToCoupledIndices 
      = [](long iBte1, long iBte2, [[maybe_unused]] const Particle &p1, [[maybe_unused]] const Particle &p2){ 
          return std::make_tuple(iBte1, iBte2); }; 

  // we save the diagonal matrix element in a dedicated vector
  std::shared_ptr<VectorBTE> internalDiagonal, internalDiagonalUmklapp, internalDiagonalNormal;
  // the scattering matrix, initialized if highMemory==true
  ParallelMatrix<double> theMatrix;

  int numStates; // number of Bloch states (i.e. the size of theMatrix)
  int numCalculations;  // number of "Calculations", i.e. number of temperatures and
  // chemical potentials on which we compute scattering.
  int dimensionality_;

  // number of states for shifting to the coupled matrix
  int numElStates; // this will never be used unless it's a coupled matrix
  // however, it still needs to be here because the coupled
  // shift incides function unfortunately also has to be here. For the case of the
  // ph matrix, we call phScattering using the BasePh object -- even if the
  // matrix supplied is a Coupled matrix, the base class's function is called.

  // this is to exclude problematic Bloch states, e.g. acoustic phonon modes
  // at gamma, which have zero frequencies and thus have several non-analytic
  // behaviors (e.g. Bose--Einstein population=infinity). We set to zero
  // terms related to these states.
  std::vector<int> excludeIndices;

  /** Method that actually computes the scattering matrix.
   * Pure virtual function: needs an implementation in every subclass.
   */
  virtual void builder(std::shared_ptr<VectorBTE> linewidth,
                       std::vector<VectorBTE> &inPopulations,
                       std::vector<VectorBTE> &outPopulations) = 0;

  /** Returns a vector of pairs of wavevector indices to iterate over during
   * the construction of the scattering matrix.
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
  std::vector<std::tuple<std::vector<int>, int>> getIteratorWavevectorPairs(
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
  void degeneracyAveragingLinewidths(std::shared_ptr<VectorBTE> linewidth);

  /** Function to precompute particle populations before scattering rates
   * are calculated.
   * @param Bandstructure: bandstructure of the particle species
   * @return MatrixXd occupationFactors: contains the occupations for all states
   */
  Eigen::MatrixXd precomputeOccupations(BaseBandStructure &bandStructure);

  /** Method which for a phonon bandstructure returns the indices to be
  * discarded in a phonon calculation due to very low phonon frequencies
  * @param bandStructure: bandstructure to compute indices for
  * @return excludeIndices: the irr indices to be excluded in ph calculations
  */
  std::vector<int> getExcludeIndices(BaseBandStructure& bandStructure);

  /* Set the use case of the scattering matrix into a class enum. 
  * Choices are 
  * we compute and store in memory the scattering matrix and the diagonal
  * we compute the action of the scattering matrix on the in vector, returning outVec = sMatrix*vector
  * we compute only the linewidths
  * @param linewidths pointer to a vectorBTE containing the internal diagonal of the scattering matrix 
  * @param inPopulations particle population the scattering matrix acts on 
  * @param outPopulations particle population Smatrix * inPopulation 
  */
  void setMatrixCase(std::shared_ptr<VectorBTE> linewidth, 
    std::vector<VectorBTE> &inPopulations,
    std::vector<VectorBTE> &outPopulations); 

  /** Method to add scattering rate to matrix and linewidth containers
   * @param context the context for the calculation 
   * @param rate the scattering rate being added 
   * @param iBte1 the BTE index of the first state
   * @param iBte2 the BTE index of the second state
   * @param p1 the particle type associated with iBte1
   * @param p2 the particle type associated with iBte2
   * @param inPopulations 
   * @param outPopulations
   */
  void addRateToMatrix(const Context &context, double linewidthRate, double matrixRate, 
                                      int iCalc, int is1, int is2Irr, int iBte1, int iBte2, 
                                      const Particle &p1, const Particle &p2, 
                                      const Eigen::Matrix3d &rotation, 
                                      std::shared_ptr<VectorBTE> linewidth, 
                                      const std::vector<VectorBTE> &inPopulations,
                                      std::vector<VectorBTE> &outPopulations); 

  /** A function to fix the linewidths to agree with the off diagonal elements, to enforce finding the
   * special eigenvectors. 
   */
  void enforceDetailedBalance();

  /** Replace the linewidths of the scatterng matrix with the supplied VectorBTE values.
   **/
  void replaceMatrixLinewidths();

  /** Returns a tuple of final and initial particles for a give state
  * @param iBte1, iBte2: the bte indices used to index this state
  */
  std::tuple<BaseBandStructure*,BaseBandStructure*> getStateBandStructures(const BteIndex& iBte1,
                                                                    const BteIndex& iBte2);

  /** If we have a coupled scattering matrix, we need to shift the matrix element indices
   * back to those associated with ph and el bandstructures before accessing their quantities
  * @param iBte1, iBte2: the bte indices used to index the global matrix 
  * @param p1, p2: the particle types indicating the quadrant of this element
  * @return: a tuple containing the bandstructure relevant indices
  */
  std::tuple<int,int> coupledToBandStructureIndices(const int& iBte1, const int& iBte2,
                                                    const Particle& p1, const Particle& p2);

  /** Generic function to add UN times to their containers 
   * @param iCalc the calculation index labeling T and mu
   * @param iBte the index of the linewidth container 
   * @param rate scattering rate of this process
   * @param crysPoints list of Points objects which will be used to calculate if is U process. 
   *          NOTE: this must be a list in the same order as the momentumConservation expects them... 
   * @param momentumConsnExpr lambda function returning the expression for the final wavevector to be folded 
   */
  template <size_t n>
  void addUNRates(int iCalc, int iBte, double rate, 
                                    const std::array<Point, n> &crysPoints, auto momentumConsExpr) {

    if(outputUNTimes) {
      // Note : for the purpose of folding bzToWs for points, 
      // the bandstructure we use doesn't matter, only the points object

      // get vectors in ws cell 
      std::array<Eigen::Vector3d, n> wsVectors;  
      for(size_t ipt = 0; ipt < n; ipt++) {
        Eigen::Vector3d wvCart = crysPoints[ipt].getCoordinates(Points::cartesianCoordinates); 
        Eigen::Vector3d wvWS = outerBandStructure.getPoints().bzToWs(wvCart, Points::cartesianCoordinates); 
        wsVectors[ipt] = wvWS; 
      }
      
      // check if this process is umklapp
      Eigen::Vector3d wvFinalCart = std::apply(momentumConsExpr, wsVectors); 
      Eigen::Vector3d wvFinalFold = outerBandStructure.getPoints().bzToWs(wvFinalCart, Points::cartesianCoordinates);
      if(abs((wvFinalCart-wvFinalFold).norm()) > 1e-6) {
        internalDiagonalUmklapp->operator()(iCalc, 0, iBte) += rate;
      } else {
        internalDiagonalNormal->operator()(iCalc, 0, iBte) += rate;
      }
    }
  }
  
  // friend functions for scattering
  friend void addBoundaryScattering(ScatteringMatrix &matrix, Context &context,
    std::vector<VectorBTE> &inPopulations,
    std::vector<VectorBTE> &outPopulations,
    BaseBandStructure &outerBandStructure,
    std::shared_ptr<VectorBTE> linewidth);

};

#endif
