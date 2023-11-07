#ifndef EL_SCATTERING_MATRIX_H
#define EL_SCATTERING_MATRIX_H

#include "electron_h0_wannier.h"
#include "interaction_elph.h"
#include "phonon_h0.h"
#include "base_el_scattering_matrix.h"
#include "vector_bte.h"

/** This class describes the construction of the electron scattering matrix.
 * The most important part is the assembly of the electron-phonon scattering.
 * We also include boundary scattering effects.
 */
class ElScatteringMatrix : virtual public BaseElScatteringMatrix {
public:
  /** Default constructor
   *
   * @param context_: object with user parameters for this calculation
   * @param statisticsSweep_: object with values for temperature and chemical
   * potential
   * @param innerBandStructure_: this is the band structure used for the
   * integration of lifetimes/scattering rates
   * @param outerBandStructure_: this is the band structure used to define on
   * which points to compute the lifetimes/scattering rates. For transport
   * properties outer=inner, but may differ e.g. when computing lifetimes on a
   * path
   * @param h0: phonon hamiltonian used to compute phonon energies and
   * eigenvectors.
   * @param couplingElPhWan_: object with the electron-phonon coupling.
   */
  ElScatteringMatrix(Context &context_, StatisticsSweep &statisticsSweep_,
                     BaseBandStructure &innerBandStructure_,
                     BaseBandStructure &outerBandStructure_, PhononH0 &h0,
                     InteractionElPhWan *couplingElPhWan_ = nullptr);

  /** Function to return the momentum relaxation times specifically */
  VectorBTE getSingleModeMRTimes();

protected:

  InteractionElPhWan *couplingElPhWan;
  PhononH0 &phononH0;

  /** Function with the detailed calculation of the scattering matrix.
   *
   * Note: this function is computing the symmetrized scattering matrix
   * $\tilde{\Omega}$.
   * As a result, only use this with the appropriately symmetrized BTE
   *
   * @param linewidth
   * @param inPopulations
   * @param outPopulations
   */
  void builder(std::shared_ptr<VectorBTE> linewidth,
               std::vector<VectorBTE> &inPopulations,
               std::vector<VectorBTE> &outPopulations) override;

  // TODO write docstrings for these
  // friend functions for adding scattering rates,
  // these live in el_scattering.cpp

  friend void addElPhScattering(BaseElScatteringMatrix &matrix, Context &context,
                       std::vector<VectorBTE> &inPopulations,
                       std::vector<VectorBTE> &outPopulations,
                       int &switchCase,
                       std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                       Eigen::MatrixXd &innerFermi, //Eigen::MatrixXd &outerBose,
                       BaseBandStructure &innerBandStructure,
                       BaseBandStructure &outerBandStructure,
                       PhononH0 &phononH0,
                       InteractionElPhWan *couplingElPhWan,
                       std::shared_ptr<VectorBTE> linewidth);

  friend void addDragTerm(BaseElScatteringMatrix &matrix, Context &context,
                  std::vector<std::tuple<std::vector<int>, int>> kqPairIterator,
                  int dragTermType,
                  ElectronH0Wannier* electronH0,
                  InteractionElPhWan *couplingElPhWan,
                  BaseBandStructure &innerBandStructure,
                  BaseBandStructure &outerBandStructure, std::shared_ptr<VectorBTE> linewidth);

  friend void addChargedImpurityScattering(BaseElScatteringMatrix &matrix, Context &context,
                       std::vector<VectorBTE> &inPopulations,
                       std::vector<VectorBTE> &outPopulations,
                       int &switchCase,
                       std::vector<std::tuple<std::vector<int>, int>> kPairIterator,
                       BaseBandStructure &innerBandStructure,
                       BaseBandStructure &outerBandStructure,
                       std::shared_ptr<VectorBTE> linewidth);

};

#endif
