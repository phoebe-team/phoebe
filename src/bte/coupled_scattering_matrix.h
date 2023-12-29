#ifndef COUPLED_SCATTERING_MATRIX_H
#define COUPLED_SCATTERING_MATRIX_H

#include "electron_h0_wannier.h"
#include "phonon_h0.h"
#include "interaction_3ph.h"
#include "interaction_elph.h"
#include "base_el_scattering_matrix.h"
#include "base_ph_scattering_matrix.h"
#include "coupled_vector_bte.h"

/** class representing the combined scattering matrix.
 * This class contains the logic to compute the combined scattering matrix.
 * For a coupled BTE solve.
 * The parent class ScatteringMatrix instead contains the logic for managing
 * the operations with distribution vectors.
 *    electron-self | electron-drag
 *  -----------------------------
 *    phonon-drag   | phonon-self
 *
 */
class CoupledScatteringMatrix : virtual public BaseElScatteringMatrix,
                                virtual public BasePhScatteringMatrix {

 public:

// TODO update all these comments

  /** Default constructor
   * @param context: the user-initialized variables.
   * @param statisticsSweep: the object containing the information on the
   * temperatures to be used in the calculation.
   * @param innerBandStructure: this is the band structure object used for
   * integrating the sum over final state wavevectors.
   * @param outerBandStructure: this is the bandStructure object used for
   * integrating the sum over initial state wavevectors.
   * @param coupling3Ph: a pointer to the class handling the 3-phonon
   * interaction calculation.
   * @param couplingElPh: a pointer to the class handling the el-ph
   * interaction calculation.
   * @param PhononH0: the object used for constructing phonon energies.
   * @param ElectronH0: the object used for constructing electron energies.
   *
   * Note: For transport calculations inner=outer.
   * Other scattering matrices allow for the possibility that they aren't equal,
   * but this matrix will only be used for transport.
   */
 CoupledScatteringMatrix(Context &context_, StatisticsSweep &statisticsSweep_,
                    BaseBandStructure &innerBandStructure_,
                    BaseBandStructure &outerBandStructure_,
                    Interaction3Ph *coupling3Ph_ = nullptr,
                    InteractionElPhWan *couplingElPh_ = nullptr,
                    ElectronH0Wannier *electronH0_ = nullptr,
                    PhononH0 *phononH0_ = nullptr);

  // TODO we will need to override the simple scattering matrix version of this function
  // as this one will need to behave differently than the others.
  // we may want to output each kind of linewidths, etc, for testing?
  /** Outputs the quantity to a json file.
   * @param outFileName: string representing the name of the json file
   */
  //void outputToJSON(const std::string &outFileName);

 // TODO check on the functions that symmetrize this matrix's components

  BaseBandStructure* getPhBandStructure();
  BaseBandStructure* getElBandStructure();

 protected:

  Interaction3Ph *coupling3Ph;
  InteractionElPhWan *couplingElPh;

  ElectronH0Wannier *electronH0;
  PhononH0 *phononH0;

  //int numElStates;
  int numPhStates;

  /** convert the phonon part of the coupled scattering matrix to Omega format.
  * this is needed because the electron part is already Omega by default,
  * but the phonon scattering rate functions output A.
  */
  void phononOnlyA2Omega();

// TODO can we remove entirely this linewidth thing? it's inconvenient
  // implementation of the scattering matrix
  void builder(std::shared_ptr<VectorBTE> linewidth,
               std::vector<VectorBTE> &inPopulations,
               std::vector<VectorBTE> &outPopulations);

  /** A function to get the wavevector index given the index of an element of theMatrix
  * @param bandstructure: the bandstructure object to which the state corresponds
  * @param iMat: the matrix element index to be converted
  * @return : wavevectorIndex of this iMat element
  */
  int bteStateToWavevector(BteIndex& iMat, BaseBandStructure& bandStructure);

  /** A function to generate the wavevector index pairs for which scattering rates are computed.
  * The coupled scattering matrix is a special case. It will always be in memory, and pairs will
  * always be between final and initial bandstructures which are either the same, or el-ph pairs.
  * Therefore, we always distribute over the states of the parallel matrix which are local,
  * and return these kq pairs. Both arguments here are unused, but left in place because this function
  * should override the scattering matrix one, to ensure that function isn't called for this Smatrix.
  * @param switchCase: dummy variable
  * @param rowMajor: dummy variable
  * @return : a vector containing the four necessary iterators, over k pairs, kq pairs, or q pairs
  *     needed to compute the scattering rates.
  */
  std::vector<std::vector<std::tuple<std::vector<int>, int>>>
  getIteratorWavevectorPairs(const int &switchCase = 0, const bool &rowMajor = 0);

  /** Function to reweight the different quadrants of the matrix
   * coupled matrix to account for spin dengeneracy */
  void reweightQuadrants();

  // friend functions for adding scattering rates
  // see respective header files for more details
  friend void addPhPhScattering(BasePhScatteringMatrix &matrix, Context &context,
                  std::vector<VectorBTE> &inPopulations,
                  std::vector<VectorBTE> &outPopulations,
                  int &switchCase,
                  std::vector<std::tuple<std::vector<int>, int>> qPairIterator,
                  Eigen::MatrixXd &innerBose, Eigen::MatrixXd &outerBose,
                  BaseBandStructure &innerBandStructure,
                  BaseBandStructure &outerBandStructure,
                  PhononH0* phononH0,
                  Interaction3Ph *coupling3Ph,
                  VectorBTE *linewidth);

  friend void addIsotopeScattering(BasePhScatteringMatrix &matrix, Context &context,
                  std::vector<VectorBTE> &inPopulations,
                  std::vector<VectorBTE> &outPopulations, int &switchCase,
                  std::vector<std::tuple<std::vector<int>, int>> qPairIterator,
                  Eigen::MatrixXd &innerBose, Eigen::MatrixXd &outerBose,
                  BaseBandStructure &innerBandStructure,
                  BaseBandStructure &outerBandStructure,
                  VectorBTE *linewidth);

  friend void addPhElScattering(BasePhScatteringMatrix& matrix, Context& context,
                  BaseBandStructure& phBandStructure,
                  ElectronH0Wannier* electronH0,
                  InteractionElPhWan* couplingElPhWan,
                  std::shared_ptr<VectorBTE> linewidth);

  friend void addElPhScattering(BaseElScatteringMatrix &matrix, Context &context,
                  std::vector<VectorBTE> &inPopulations,
                  std::vector<VectorBTE> &outPopulations,
                  int &switchCase,
                  std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                  Eigen::MatrixXd &innerFermi, Eigen::MatrixXd &outerBose,
                  BaseBandStructure &innerBandStructure,
                  BaseBandStructure &outerBandStructure,
                  PhononH0 &phononH0,
                  InteractionElPhWan *couplingElPhWan,
                  VectorBTE *linewidth);

  friend void addChargedImpurityScattering(BaseElScatteringMatrix &matrix, Context &context,
                  std::vector<VectorBTE> &inPopulations,
                  std::vector<VectorBTE> &outPopulations,
                  int &switchCase,
                  std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                  BaseBandStructure &innerBandStructure,
                  BaseBandStructure &outerBandStructure,
                  std::shared_ptr<VectorBTE> linewidth);

  friend void addDragTerm(CoupledScatteringMatrix &matrix, Context &context,
                  std::vector<std::tuple<std::vector<int>, int>> kqPairIterator,
                  int dragTermType,
                  ElectronH0Wannier* electronH0,
                  InteractionElPhWan *couplingElPhWan,
                  BaseBandStructure &innerBandStructure,
                  BaseBandStructure &outerBandStructure);

  friend void phononElectronAcousticSumRule(CoupledScatteringMatrix &matrix,
                  Context& context,
                  std::shared_ptr<CoupledVectorBTE> phElLinewidths,
                  BaseBandStructure& elBandStructure,
                  BaseBandStructure& phBandStructure);
				
};

#endif
