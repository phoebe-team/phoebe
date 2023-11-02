#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <complex>
#include <cmath>
//
// mathematical constants
//
const double pi = 3.14159265358979323846;
const double sqrtPi = sqrt(3.14159265358979323846);
const double sqrtTwo = sqrt(2.);
const double twoPi = 2. * pi;
const double fourPi = 2. * pi;
const double one = 1.;
const double zero = 0.;
const std::complex<double> complexZero = {0., 0.}; /* NOLINT */
const std::complex<double> complexOne = {1., 0.}; /* NOLINT */
const std::complex<double> complexI = {0., 1.}; /* NOLINT */
//
// Physical constants
//
const double speedLightSi = 299792458;
const double electronSi = 1.602176487e-19; // electron charge in Coulomb
const double hPlanckSi = 6.62607015e-34;
const double hBarSi = hPlanckSi / twoPi;
const double electronVoltSi = 1.6021766208e-19;
const double hartreeSi = 4.359744650e-18;
const double bohrRadiusSi = 0.52917721067e-10;
const double rydbergSi = hartreeSi / 2.;
const double timeRy = hBarSi / rydbergSi; // note: prefer using timeAuToFs
// as this might miss some twoPi factors
const double kBoltzmannSi = 1.38064852e-23;
const double kBoltzmannRy = kBoltzmannSi / rydbergSi;
const double electronMassSi = 9.10938215e-31; // in Kg
const double amuSi = 1.660538782e-27;         // in Kg

const double e2 = 2.;

//
// conversion
//
const double temperatureAuToSi = 1. / kBoltzmannRy;
const double energyRyToEv = rydbergSi / electronVoltSi;
const double distanceAngToSi = 1.0e-10;
const double distanceRyToSi = bohrRadiusSi;
const double distanceBohrToCm = bohrRadiusSi * 100.;
const double distanceBohrToMum = bohrRadiusSi * 1000000.;
const double timeAuToFs = timeRy * 1e15; // * twoPi * twoPi;
const double energyRyToFs = timeAuToFs * twoPi; // to convert linewidths!
const double mevToPs = hPlanckSi * 1e15 / electronVoltSi;
const double distanceBohrToAng = bohrRadiusSi / distanceAngToSi;
const double massAmuToRy = amuSi / electronMassSi / 2.;
const double massRyToAmu = 1. / massAmuToRy;

const double TeraHertzAu = hPlanckSi / twoPi / hartreeSi * 1.0e+12;

const double ryToTHz = 1. / TeraHertzAu / 2. / twoPi;
const double ryToCmm1 = 1.0e10 * ryToTHz / speedLightSi;

// velocity tested for phonons
const double velocityRyToSi = distanceRyToSi * rydbergSi / hBarSi;

//
// units for transport coefficients
//
const double thConductivityAuToSi = kBoltzmannSi * rydbergSi / bohrRadiusSi / hBarSi;
const double viscosityAuToSi = 1./rydbergSi * pow(hBarSi,2) * 1./pow(bohrRadiusSi,3) * 1./timeRy;
const double elConductivityAuToSi = electronSi * electronSi / hBarSi / bohrRadiusSi;
const double mobilityAuToSi = electronSi / hBarSi * bohrRadiusSi * bohrRadiusSi;
const double thermopowerAuToSi = kBoltzmannRy / electronSi * rydbergSi;

#endif
