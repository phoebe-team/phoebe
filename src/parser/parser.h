#ifndef PARSER_H
#define PARSER_H

#include "context.h"
#include "electron_h0_fourier.h"
#include "electron_h0_wannier.h"
#include "phonon_h0.h"
#include "phonopy_input_parser.h"
#include "jdftx_input_parser.h"
#include "qe_input_parser.h"
#include <string>

/** Class used to make decisions about which parser to call
 * depending on which options were provided as input.
 */
class Parser {
public:
  /** parsing of force constants.
   * @param context: the object containing the user input.
   * @return Crystal: crystal is the object describing the crystal structure.
   * @return PhononH0: the object containing the force Constants and the
   * functionality to compute the phonon energies.
   */
  static std::tuple<Crystal, PhononH0> parsePhHarmonic(Context &context);

  /** parsing of electronic band structure for Fourier interpolation.
   * This class parses the XML file of Quantum ESPRESSO, which should contain
   * the band structure computed on a uniform grid of k-points.
   * @param context: the object containing the user input.
   * @return Crystal: the object describing the crystal structure
   * @return ElectronH0Fourier: the object containing the band structure on
   * the coarse grid of points, with all the infrastructure necessary to
   * interpolate it on finer grids of k-points.
   */
  static std::tuple<Crystal, ElectronH0Fourier>
  parseElHarmonicFourier(Context &context);

  /** parsing of electronic band structure for Wannier interpolation.
   * This class parses the _tb.dat file from Wannier90, which is created
   * when Wannier90 is run with the parameter write_tb = true
   * @param context: the object containing the user input.
   * @return Crystal: the object describing the crystal structure
   * @return ElectronH0Wannier: the object containing the ground state
   * electronic Hamiltonian in the Wannier representation, and capable of
   * interpolating the band structure on a fine grid of k-points.
   */
  static std::tuple<Crystal, ElectronH0Wannier>
  parseElHarmonicWannier(Context &context, Crystal *inCrystal = nullptr);
};

#endif
