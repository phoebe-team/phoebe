#ifndef PARTICLE_H
#define PARTICLE_H

#include "eigen.h"

/////

/** Determines whether we are dealing with phonons or electrons.
 *
 * This class decides whether we are dealing with phonons or electrons.
 * Additionally, it contains the functionality to compute equilibrium
 * occupation numbers, i.e. Fermi-Dirac or Bose-Einstein distributions, their
 * derivatives w.r.t. temperature or energy, and factors N(N+1) or F(1-F).
 *
 * @ref Statistics
 */
class Particle {
public:
  static const int electron = -1;
  static const int phonon = -2;

  /** Constructor of Statistics. Saves which kind of particle we are dealing
   * with.
   * @param statistics: an integer equal to Statistics::electron or
   * Statistics::phonon.
   */

  explicit Particle(int kind_);

  /** Returns either a Bose--Einstein or a Fermi--Dirac distribution,
   * depending on the value of "statistics".
   * @param energy: value of quasiparticle energy.
   * @param temperature: value of temperature.
   * @param chemicalPotential: 0 by default, set a value for electrons.
   * @return n: either the Bose or Fermi population.
   */
  double getPopulation(const double &energy, const double &temperature,
                       const double &chemicalPotential = 0.) const;

  /** Returns dn/dT, with T temperature, and n being either a Bose--Einstein
   * or a Fermi--Dirac distribution, depending on the value of "statistics"
   *   NOTE: this is off by a factor of kB!
   *   dn/dT  = n(n+/-1) * energy / kB T^2
   *   Here, we are returning something which is (kB * T)^2
   *   It might be good to change this, but it's used other places
   *   and this would require careful fixing.
   * @param energy: value of quasiparticle energy.
   * @param temperature: value of temperature.
   * @param chemicalPotential: 0 by default, set a value for electrons.
   * @param symmetrize: false by default, if true rescales dndt by a factor
   * 1/sqrt(f(1-f)) or 1/sqrt(n(1+n)). Used for the symmetrized BTE.
   * @return dndT: derivative wrt temperature of the equilibrium distribution
   */
  double getDndt(const double &energy, const double &temperature,
                 const double &chemicalPotential = 0.,
                 const bool &symmetrize = false) const;

  /** Returns dn/dE, with E the particle energy, and n being either a
   * Bose or Fermi distribution, depending on the value of "statistics".
   * @param energy: value of quasiparticle energy.
   * @param temperature: value of temperature.
   * @param chemicalPotential: 0 by default, set a value for electrons.
   * @param symmetrize: false by default, if true rescales dndt by a factor
   * 1/sqrt(f(1-f)) or 1/sqrt(n(1+n)). Used for the symmetrized BTE.
   * @return dndE: derivative wrt QP energy of the equilibrium distribution.
   */
  double getDnde(const double &energy, const double &temperature,
                 const double &chemicalPotential = 0.,
                 const bool &symmetrize = false) const;

  /** Returns bose(bose+1) for bosons, fermi(1-fermi) for fermions.
   * @param energy: value of quasiparticle energy.
   * @param temperature: value of temperature.
   * @param chemicalPotential: 0 by default, set a value for electrons.
   * @return popPopPm1: bose(bose+1) for bosons, fermi(1-fermi) for fermions.
   */
  double getPopPopPm1(const double &energy, const double &temperature,
                      const double &chemicalPotential = 0.) const;

  /** Method to check if the particle is a fermion.
   */
  bool isFermi() const;

  /** Method to check if the particle is a boson.
   */
  bool isBose() const;

  /** Method to check if the particle is an electron.
   */
  bool isElectron() const;

  /** Method to check if the particle is a phonon.
   */
  bool isPhonon() const;

  int getParticleKind() const;

private:
  static const int fermi = 0;
  static const int bose = 1;

  // identifies whether the instance is a fermion of a boson
  int statistics;
  // identifies whether we have an electron or a phonon
  // Note: we keep particle != statistics because we might add particles
  int kind;
};

#endif
