#ifndef PHEL_SCATTERING_MATRIX_H
#define PHEL_SCATTERING_MATRIX_H

#include "electron_h0_wannier.h"
#include "phonon_h0.h"
#include "base_ph_scattering_matrix.h"
#include "vector_bte.h"
#include "interaction_elph.h"

// TODO realistically this should be just absorbed
// as a diagonal only case for ph scattering matrix!
/** This class describes the construction of the electron scattering matrix.
 * The most important part is the assembly of the electron-phonon scattering.
 * We also include boundary scattering effects.
 */
class PhElScatteringMatrix : public BasePhScatteringMatrix {
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
   */
  PhElScatteringMatrix(Context &context_, StatisticsSweep &statisticsSweep_,
                       //BaseBandStructure &elBandStructure_,
                       BaseBandStructure &phBandStructure_,
                       PhononH0 *phononH0_, 
                       ElectronH0Wannier *electronH0_);

  /** Copy constructor
   * @param that: object to be copied
   */
  PhElScatteringMatrix(const PhElScatteringMatrix &that);

  /** Copy assignment
   *
   * @param that: object to be copied
   * @return a copy of ElScatteringMatrix
   */
  PhElScatteringMatrix &operator=(const PhElScatteringMatrix &that);

protected:

  ElectronH0Wannier *electronH0;
  PhononH0* phononH0; 

  void builder(std::shared_ptr<VectorBTE> linewidth, std::vector<VectorBTE> &inPopulations,
               std::vector<VectorBTE> &outPopulations) override;

  // to prevent mistakes in which these two outer and inner BS could
  // be accidentally swapped
  BaseBandStructure& getPhBandStructure() { return outerBandStructure; };
  BaseBandStructure& getElBandStructure() { return innerBandStructure; };

  friend void addPhElScattering(BasePhScatteringMatrix &matrix, Context &context,
                BaseBandStructure &phBandStructure,
                BaseBandStructure &elBandStructure,
                StatisticsSweep &statisticsSweep, 
                InteractionElPhWan &couplingElPhWan,
                std::shared_ptr<VectorBTE> linewidth);

};

#endif
