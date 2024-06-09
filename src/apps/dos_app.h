#ifndef DOS_APP_H
#define DOS_APP_H

#include "app.h"
#include <string>

class PhononDosApp : public App {
public:
  void run(Context &context) override;
  void checkRequirements(Context &context) override;
};

class ElectronWannierDosApp : public App {
public:
  void run(Context &context) override;
  void checkRequirements(Context &context) override;
};

class ElectronFourierDosApp : public App {
public:
  void run(Context &context) override;
  void checkRequirements(Context &context) override;
};

/**
 * Helper function to output dos to a json file
 * @param energies: list of energies to write to file 
 * @param dos: list of dos values to write to file 
 * @param particle: electron or phonon particle object 
 * @param outFileName: name of the dos .json file 
 */
void outputDOSToJSON(std::vector<double> energies, std::vector<double> dos,
          Particle& particle, const std::string &outFileName);

#endif
