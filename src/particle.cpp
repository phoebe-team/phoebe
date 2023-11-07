#include "particle.h"

#include "exceptions.h"

Particle::Particle(int kind_) {
  if (kind_ == phonon) {
    statistics = bose;
    kind = phonon;
  } else if (kind_ == electron) {
    statistics = fermi;
    kind = electron;
  } else {
    Error("Wrong initialization of Particle", 1);
  }
}

bool Particle::isFermi() const {
  if (statistics == fermi) {
    return true;
  } else {
    return false;
  }
}

bool Particle::isBose() const {
  if (statistics == bose) {
    return true;
  } else {
    return false;
  }
}

bool Particle::isElectron() const {
  if (kind == electron) {
    return true;
  } else {
    return false;
  }
}

bool Particle::isPhonon() const {
  if (kind == phonon) {
    return true;
  } else {
    return false;
  }
}

double Particle::getPopulation(const double &energy, const double &temperature,
                               const double &chemicalPotential) const {
  double population = 0.;

  if (temperature <= 0.) {
    if (energy <= chemicalPotential) {
      population = 1.;
    }
    return population;
  }

  double y = (energy - chemicalPotential) / temperature;

  // note: since the value of the exponential might be noisy for small/large
  // numbers, we make sure that the population is >=0 (or <=1 for fermions)

  if (statistics == bose) {
    population = 1. / (exp(y) - 1);
    if (population < 0.) {
      population = 0.;
    }
  } else {
    population = 1. / (exp(y) + 1);

    if (population < 0.) {
      population = 0.;
    }
    if (population > 1.) {
      population = 1.;
    }
  }
  return population;
}

// NOTE from Jenny: this is off by a factor of kB!
// dn/dT  = n(n+/-1) * energy / kB T^2
//   Here, we are returning something which is (kB * T)^2, as here T = kBT
//   I am afraid to change this now because it's used in several other places...
double Particle::getDndt(const double &energy, const double &temperature,
                         const double &chemicalPotential,
                         const bool &symmetrize) const {

  double x = getPopPopPm1(energy, temperature, chemicalPotential);
  if (symmetrize) x = sqrt(x);
  double y = energy - chemicalPotential;
  double dndt = x * y / temperature / temperature;
  return dndt;
}

double Particle::getDnde(const double &energy, const double &temperature,
                         const double &chemicalPotential,
                         const bool &symmetrize) const {
  double x = getPopPopPm1(energy, temperature, chemicalPotential);
  if (symmetrize) x = sqrt(x);
  double dnde = -x / temperature;
  return dnde;
}

double Particle::getPopPopPm1(const double &energy, const double &temperature,
                              const double &chemicalPotential) const {
  double x = energy - chemicalPotential;
  double arg = x / 2. / temperature;
  if (statistics == bose) {
    x = sinh(arg); // numerically more stable than n(n+1)
  } else {
    x = cosh(arg); // numerically more stable than f(1-f)
  }
  x = 0.25 / x / x;
  return x;
}

int Particle::getParticleKind() const { return kind; }
