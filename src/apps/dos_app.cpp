#include <fstream>
#include <iostream>
#include <string>
#include "dos_app.h"
#include "exceptions.h"
#include "constants.h"
#include "delta_function.h"
#include "electron_h0_fourier.h"
#include "utilities.h"
#include "qe_input_parser.h"

// Compute the DOS with the tetrahedron method
void PhononDosApp::run(Context & context) {
	std::cout << "Starting phonon DoS calculation" << std::endl;

	// Read the necessary input files
	auto [crystal, phononH0] = QEParser::parsePhHarmonic(context);

	// first we make compute the band structure on the fine grid
	FullPoints fullPoints(crystal, context.getQMesh());
	bool withVelocities = false;
	bool withEigenvectors = false;
	FullBandStructure fullBandStructure = phononH0.populate(
			fullPoints, withVelocities, withEigenvectors);

	// Form tetrahedra and fill them with eigenvalues
	TetrahedronDeltaFunction tetrahedra(fullBandStructure);

	double minEnergy = context.getDosMinEnergy();
	double maxEnergy = context.getDosMaxEnergy();
	double deltaEnergy = context.getDosDeltaEnergy();
	long numEnergies = (maxEnergy - minEnergy) / deltaEnergy + 1;
	std::vector<double> energies(numEnergies);
	for ( long i=0; i<numEnergies; i++ ) {
		energies[i] = i * deltaEnergy;
	}

	// Calculate phonon density of states (DOS) [1/Ry]
	std::vector<double> dos(numEnergies, 0.); // phonon DOS initialized to zero
	for ( long i=0; i<numEnergies; i++ ) {
		dos[i] += tetrahedra.getDOS(energies[i]);
	}

	// Save phonon DOS to file
	std::ofstream outfile("./phonon_dos.dat");
	outfile << "# Phonon density of states: frequency[Cmm1], Dos[1/Ry]\n";
	for ( long i=0; i<numEnergies; i++ ) {
		outfile << energies[i] * ryToCmm1 << "\t" << dos[i] << "\n";
	}
	std::cout << "Phonon DoS computed" << std::endl;
}

// Compute the Electron DOS with tetrahedron method and Fourier interpolation
void ElectronWannierDosApp::run(Context & context) {
	std::cout << "Starting electronic DoS calculation" << std::endl;

	// Read the necessary input files

	auto [crystal, h0] = QEParser::parseElHarmonicWannier(context);

	// first we make compute the band structure on the fine grid
	FullPoints fullPoints(crystal, context.getKMesh());
	bool withVelocities = false;
	bool withEigenvectors = false;
	FullBandStructure fullBandStructure = h0.populate(
			fullPoints, withVelocities, withEigenvectors);

	// Form tetrahedra and fill them with eigenvalues
	TetrahedronDeltaFunction tetrahedra(fullBandStructure);

	double minEnergy = context.getDosMinEnergy();
	double maxEnergy = context.getDosMaxEnergy();
	double deltaEnergy = context.getDosDeltaEnergy();
	long numEnergies = (maxEnergy - minEnergy) / deltaEnergy + 1;
	std::vector<double> energies(numEnergies);
	for ( long i=0; i<numEnergies; i++ ) {
		energies[i] = i * deltaEnergy + minEnergy;
	}

	// Calculate phonon density of states (DOS) [1/Ry]
	std::vector<double> dos(numEnergies, 0.); // phonon DOS initialized to zero
	for ( long i=0; i<numEnergies; i++ ) {
		dos[i] += tetrahedra.getDOS(energies[i]);
	}

	// Save phonon DOS to file
	std::ofstream outfile("./electron_dos.dat");
	outfile << "# Electronic density of states: energy[eV], Dos[1/Ry]\n";
	for ( long i=0; i<numEnergies; i++ ) {
		outfile << energies[i] * energyRyToEv << "\t"
				<< dos[i]/energyRyToEv << "\n";
	}
	std::cout << "Electronic DoS computed" << std::endl;
}

// Compute the Electron DOS with tetrahedron method and Fourier interpolation
void ElectronFourierDosApp::run(Context & context) {
	std::cout << "Starting electronic DoS calculation" << std::endl;

	// Read the necessary input files

	auto [crystal, h0] = QEParser::parseElHarmonicFourier(context);

	// first we make compute the band structure on the fine grid
	FullPoints fullPoints(crystal, context.getKMesh());
	bool withVelocities = false;
	bool withEigenvectors = false;
	FullBandStructure fullBandStructure = h0.populate(
			fullPoints, withVelocities, withEigenvectors);

	// Form tetrahedra and fill them with eigenvalues
	TetrahedronDeltaFunction tetrahedra(fullBandStructure);

	double minEnergy = context.getDosMinEnergy();
	double maxEnergy = context.getDosMaxEnergy();
	double deltaEnergy = context.getDosDeltaEnergy();
	long numEnergies = (maxEnergy - minEnergy) / deltaEnergy + 1;
	std::vector<double> energies(numEnergies);
	for ( long i=0; i<numEnergies; i++ ) {
		energies[i] = i * deltaEnergy + minEnergy;
	}

	// Calculate phonon density of states (DOS) [1/Ry]
	std::vector<double> dos(numEnergies, 0.); // phonon DOS initialized to zero
	for ( long i=0; i<numEnergies; i++ ) {
		dos[i] += tetrahedra.getDOS(energies[i]);
	}

	// Save phonon DOS to file
	std::ofstream outfile("./electron_dos.dat");
	outfile << "# Electronic density of states: energy[eV], Dos[1/Ry]\n";
	for ( long i=0; i<numEnergies; i++ ) {
		outfile << energies[i] * energyRyToEv << "\t"
				<< dos[i]/energyRyToEv << "\n";
	}
	std::cout << "Electronic DoS computed" << std::endl;
}

void PhononDosApp::checkRequirements(Context & context) {
	throwErrorIfUnset(context.getPhD2FileName(), "PhD2FileName");
	throwErrorIfUnset(context.getQMesh(), "qMesh");
	throwErrorIfUnset(context.getDosMinEnergy(), "dosMinEnergy");
	throwErrorIfUnset(context.getDosMaxEnergy(), "dosMaxEnergy");
	throwErrorIfUnset(context.getDosDeltaEnergy(), "dosDeltaEnergy");
	throwWarningIfUnset(context.getSumRuleD2(), "sumRuleD2");
}

void ElectronWannierDosApp::checkRequirements(Context & context) {
	throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
	throwErrorIfUnset(context.getQMesh(), "kMesh");
	throwErrorIfUnset(context.getDosMinEnergy(), "dosMinEnergy");
	throwErrorIfUnset(context.getDosMaxEnergy(), "dosMaxEnergy");
	throwErrorIfUnset(context.getDosDeltaEnergy(), "dosDeltaEnergy");
	std::string crystalMsg = "crystal structure";
	throwErrorIfUnset(context.getInputAtomicPositions(), crystalMsg) ;
	throwErrorIfUnset(context.getInputSpeciesNames(), crystalMsg) ;
	throwErrorIfUnset(context.getInputAtomicSpecies(), crystalMsg);
}

void ElectronFourierDosApp::checkRequirements(Context & context) {
	throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
	throwErrorIfUnset(context.getQMesh(), "kMesh");
	throwErrorIfUnset(context.getDosMinEnergy(), "dosMinEnergy");
	throwErrorIfUnset(context.getDosMaxEnergy(), "dosMaxEnergy");
	throwErrorIfUnset(context.getDosDeltaEnergy(), "dosDeltaEnergy");
	throwErrorIfUnset(context.getElectronFourierCutoff(),
			"electronFourierCutoff");
}
