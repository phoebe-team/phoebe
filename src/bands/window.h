#ifndef WINDOW_H
#define WINDOW_H

#include "context.h"
#include "eigen.h"
#include "particle.h"

/** The window class contains the logic to decide whether to keep or discard
 * a Bloch state. Mainly used by ActiveBandStructure, it contains some criteria
 * by which we decide to discard some points from the calculation.
 * Only three filters are implemented:
 * 1- no filter is applied
 * 2- population: we discard states that are scarcely (or fully) populated.
 * 3- energy: the user provides a lower an upper bound to the energies to be
 * considered.
 */
class Window {
 public:
  /** Class constructor.
   * @param context: object with user input
   * @param particle: object describing whether we have electrons or phonons.
   * @param temperatureMin: smallest value of temperature used in the
   * calculation.
   * @param temperatureMax: largest value of temperature used.
   * @param chemicalPotentialMin: smallest value of chemical potential used
   * in the calculation.
   * @param chemicalPotentialMax: largest value of chemical potential used
   * in the calculation.
   *
   * Note: temperatures and chemical potentials are used only if we are
   * using a filter on populations.
   */
  Window(Context &context, Particle &particle_, const double &temperatureMin =
  0., const double &temperatureMax = 0.,
         const double &chemicalPotentialMin = 0.,
         const double &chemicalPotentialMax = 0.);

  /** public interface to use the window, used by activeBandStructure.
   * @param: energies: a list of energies, in absolute units. They should be
   * using the same offset of the chemical potential.
   * Nota Bene: the energies should be sorted! This is typically true if we
   * are looping over the energies of a single wavevector.
   * @return filteredEnergies: the values of energies after the filter,
   * an empty vector if nothing is found.
   * @return filteredBandsExtrema: the smallest and largest band index of the
   * bands that pass the filter. We are assuming that the filtered energies
   * are contiguous in the band index. Returns an empty vector if nothing
   * goes past the filter.
   */
  std::tuple<std::vector<double>, std::vector<int>> apply(
      Eigen::VectorXd &energies);

  // Constants that identify the kind of filter to be used
  /** nothing=0 identifies the do-nothing window
   */
  static const int nothing = 0;

  /** population labels the window type looking for partially occupied states
   */
  static const int population = 1;

  /** energy labels the window type looking for states within two energy values
   */
  static const int energy = 2;

  /** the window type is provided by the user as an energy range,
   * where the values are a +/- range around mu
   */
  static const int muEnergy = 3;

  /** Returns the kind of energy filter used.
   * @return method: an integer equal to either Window::nothing,
   * Window::population, or Window::energy.
   */
  int getMethodUsed() const;
 private:
  /** particle stores whether we are working with electrons or phonons
   */
  Particle &particle;

  // parameters for window
  double temperatureMin, temperatureMax;
  double chemicalPotentialMin, chemicalPotentialMax;
  double populationThreshold = 0.;
  double minEnergy = 0., maxEnergy = 0.;

  /** variable for selecting the window type
   */
  int method;

  /** temp variable to facilitate code writing
   */
  int numBands = 0;

  // internal method to apply the window on population
  std::tuple<std::vector<double>, std::vector<int>> internalPopWindow(
      const Eigen::VectorXd &energies, const Eigen::VectorXd &popMin,
      const Eigen::VectorXd &popMax) const;
  // internal method to apply the window on energy
  std::tuple<std::vector<double>, std::vector<int>> internalEnWindow(
      const Eigen::VectorXd &energies) const;
};

#endif
