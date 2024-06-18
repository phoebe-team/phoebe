#include "io.h"
#include "mpiHelper.h"
#include "main.h"
#include <algorithm>
#include <exceptions.h>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <fstream>

// Utility to get the command line option from it's name
char *getCmdOption(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return nullptr;
}

IO::IO(int argc, char *argv[]) {
  char *inputFileName_ = getCmdOption(argv, argv + argc, "-in");
  char *outputFileName_ = getCmdOption(argv, argv + argc, "-out");
  if (inputFileName_ == nullptr) {
    Error("Must provide an input file name on command line");
  }

  { // check input file existence
    std::ifstream f(inputFileName_);
    if (not f.good()) {
      Error("Input file not found");
    }
  }

  inputFileName = inputFileName_;
  // redirect std::cout to outputFilename, if passed on command line
  if (outputFileName_ != nullptr) {
    outputFileName = outputFileName_;
    outputFile.open(outputFileName, std::ios::out);

    // Backup stream buffers of  cout
    std::streambuf* stream_buffer_cout = std::cout.rdbuf();

    // Get the stream buffer of the file
    std::streambuf* stream_buffer_file = outputFile.rdbuf();

    // Redirect cout to file
    std::cout.rdbuf(stream_buffer_file);

    // Redirect cout back to screen
    std::cout.rdbuf(stream_buffer_cout);

    // causes buffer to flush at every call of <<
    std::cout.setf( std::ios_base::unitbuf );
  }
}

IO::~IO() {
  if (outputFile.is_open()) {
    outputFile.close();
  }
}

std::string IO::getInputFileName() { return inputFileName; }

void IO::welcome() {
  if (!mpi->mpiHead())
    return;
  std::string welcomeMsg =
      "\n"
      "  8888888b.  888                        888\n"
      "  888   Y88b 888                        888\n"
      "  888    888 888                        888\n"
      "  888   d88P 88888b.   .d88b.   .d88b.  88888b.   .d88b.  \n"
      "  8888888P'  888 '88b d88''88b d8P  Y8b 888 '88b d8P  Y8b \n"
      "  888        888  888 888  888 88888888 888  888 88888888 \n"
      "  888        888  888 Y88..88P Y8b.     888 d88P Y8b.     \n"
      "  888        888  888  'Y88P'   'Y8888  88888P'   'Y8888' \n";
  std::cout << welcomeMsg << std::endl;
  std::cout << "Phoebe v" << Phoebe_VERSION_MAJOR
	    << "." << Phoebe_VERSION_MINOR << "\n" << std::endl;
}

void IO::goodbye(Context &context) {
  if (!mpi->mpiHead()) return;
  std::cout << "Exiting program.\n" << std::endl;


  // print out citations related to this run ----------------
  std::cout << "Consider citing the following references "
        << "in relation to this calculation:\n" << std::endl;

  // Phoebe citation
  std::cout << "\tPhoebe: a high-performance framework for solving \n\tphonon and electron "
               "Boltzmann transport equations.\n"
	    << "\tA. Cepellotti, J. Coulter, A. Johansson, N. S. Fedorova, B. Kozinsky.\n"
      << "\thttps://doi.org/10.1088/2515-7639/ac86f6 (2022).\n" << std::endl;

  std::vector<std::string> solvers = context.getSolverBTE();
  // Relaxons solver
  if (std::find(solvers.begin(), solvers.end(), "relaxons") != solvers.end()) {
    std::cout << "  For the use of the relaxons BTE solver:" << std::endl;
    std::cout << "\tA. Cepellotti and N. Marzari.\n" <<
    //    "\tThermal transport in crystals as a kinetic theory of relaxons.\n" <<
        "\tPhysical Review X 6, no. 4 (2016): 041013.\n" << std::endl;

    //std::cout << "Generalization of Fourier’s Law into Viscous Heat Equations" << std::endl;
    std::cout << "\tM. Simoncelli, N. Marzari, and A. Cepellotti.\n" <<
                 "\tPhysical Review X 10, no. 1 (2020): 011019.\n" << std::endl;

  }
  // iterative solver
  if (std::find(solvers.begin(), solvers.end(), "iterative") != solvers.end()) {

    std::cout << "  For the use of the iterative BTE solver:" << std::endl;
    std::cout << "\tM. Omini and A. Sparavigna. \n" <<
    //    "\tAn iterative approach to the phonon Boltzmann " <<
    //   "equation in the theory of thermal conductivity.\n" <<
        "\tPhysica B: Condensed Matter 212, no. 2 (1995): 101-112.\n" << std::endl;
  }
  // variational solver
  if (std::find(solvers.begin(), solvers.end(), "variational") != solvers.end()) {
    std::cout << "  For the use of the variational BTE solver:" << std::endl;
    std::cout << "\tG. Fugallo, M. Lazzeri, L. Paulatto, and F. Mauri.\n" <<
    //    "\tAb initio variational approach for evaluating lattice thermal conductivity.\n" <<
        "\tPhysical Review B 88, no. 4 (2013): 045430.\n" << std::endl;
  }
  // electron-phonon wannier interpolation
  if(!context.getElphFileName().empty()) {
    std::cout << "  For electron-phonon Wannier interpolation:" << std::endl;
    std::cout << "\tF. Giustino, M.L. Cohen, and S.G. Louie.\n" <<
    //    "\tElectron-phonon interaction using Wannier functions.\n" <<
        "\tPhysical Review B 76, no. 16 (2007): 165108.\n" << std::endl;
  }
  // wigner transport
  if(context.getAppName() == "phononTransport"
        || context.getAppName() == "electronWannierTransport") {
          
    std::cout << "  For the Wigner transport equations:" << std::endl;
    std::cout << "\tM. Simoncelli, N. Marzari, and F. Mauri.\n" <<
    //    "\tUnified theory of thermal transport in crystals and glasses.\n" <<
        "\tNature Physics, 15(8), (2019). pp.809-813.\n" << std::endl;
    std::cout << "\tA. Cepellotti and B. Kozinsky.\n" <<
    //    "\tInterband tunneling effects on materials transport properties using the first principles Wigner distribution.\n" <<
        "\tMaterials Today Physics 19, 100412 (2021).\n" << std::endl;
  }
  // EPA 
  // TODO add the fourier interpolation -- Boltztrap??
  if(context.getElPhInterpolation() == "epa" || context.getAppName() == "transportEpa") {
    std::cout << "  For the use of the EPA method:" << std::endl;
    std::cout << "\tS.Bang, J.Kim, D.Wee, G.Samsonidze, and B.Kozinsky.\n" <<
    //    "\tEstimation of electron-phonon coupling via moving least squares averaging: " <<
    //    "\ta method for fast-screening potential thermoelectric materials.\n" <<
        "\tAdvanced Functional Materials 29, no. 44 (2019): 1905044.\n" << std::endl;
  }
  // Wannier functions in general
  if(context.getAppName() == "electronWannierBands" ||
        context.getAppName() == "electronWannierDos" ||
        context.getAppName() == "electronLifetimes" ||
        context.getAppName() == "electronWannierTransport" || 
        context.getAppName() == "coupledTransport") {
    std::cout << "  For the use of Wannier functions and interpolation:" << std::endl;
    std::cout << "\tN. Marzari, A.A. Mostofi, J.R. Yates, I. Souza, and D. Vanderbilt.\n" <<
    //    "\tMaximally localized Wannier functions: Theory and applications.\n" <<
        "\tReviews of Modern Physics 84, no. 4 (2012): 1419.\n" << std::endl;
  }

  // Electron-phonon from ab-initio
  if(context.getAppName() == "coupledTransport" ||
      context.getAppName() == "electronLifetimes" ||
      context.getAppName() == "electronWannierTransport") {

    std::cout << "  For the use of ab-initio electron-phonon coupling:" << std::endl;
    std::cout << "\tS. Piscanec, M. Lazzeri, F. Mauri, A. C. Ferrari, and J. Robertson.\n" <<
              //    "\tKohn Anomalies and Electron-Phonon Interactions in Graphite.\n" <<
              "\tPhysical Review Letters 93, 185503 (2004)\n" << std::endl;
    // At least, I think it's this one. Subroutine elphel() in QE was
    // written by F. Mauri, but it's unclear when and for what article
  }
  // phonon isotope scattering
  if(context.getWithIsotopeScattering() && context.getAppName() == "phononTransport") {
    std::cout << "  For use of phonon-isotope scattering:" << std::endl;
    std::cout << "\tJ. Garg, N. Bonini, B. Kozinsky, and N. Marzari.\n" <<
    // Role of Disorder and Anharmonicity in the Thermal Conductivity of
    // Silicon-Germanium Alloys: A First-Principles Study
             "\tPhysical Review Letters 106, 045901 (2011)\n" << std::endl;
  }
  // scattering matrix/bte symmetries
  if (context.getScatteringMatrixInMemory() && context.getUseSymmetries() && 
       (context.getAppName() == "electronWannierTransport" || 
        context.getAppName() == "phononTransport")) {

    std::cout << "  For the use of symmetries in the scattering matrix:" << std::endl;
    std::cout << "\tL. Chaput.\n" <<
              //    "\tDirect Solution to the Linearized Phonon Boltzmann Equation.\n" <<
              "\tPhysical Review Letters 110, 265506 (2013).\n" << std::endl;
  }

}

LoopPrint::LoopPrint(const std::string &task_, const std::string &step_,
                     const size_t &numSteps_) {
  if (!mpi->mpiHead())
    return;

  task = task_;
  step = step_;
  numSteps = numSteps_;

  int numRep = 10; // number of intermediate reports
  if (numSteps < numRep) {
    reportEvery = 1;
  } else {
    reportEvery = numSteps / numRep;
  }

  // note: time returns the time (in secs) elapsed from 1st Jan 1970
  initialTime = std::chrono::steady_clock::now();

  std::cout << "\n";
  std::cout << "Started " << task << " with ~" << numSteps << " " << step
            << " per MPI process." << std::endl;

  stepDigits = int(log10(numSteps)) + 1; // number of digits in numSteps
}

void LoopPrint::update(const bool &withTimeEstimate) {
  // Update prediction info for current task.
  if (!mpi->mpiHead())
    return;

  currentStep += 1;

  if (currentStep <= 2 || (currentStep + 1) % reportEvery == 0 ||
      currentStep == numSteps - 1) {

    time_point currentTime;
    currentTime = std::chrono::steady_clock::now();

    // format currentTime nicely
    char s[200];
    time_t curr = time(nullptr);
    struct tm *p = localtime(&curr);
    strftime(s, 200, "%F, %T", p);

    time_delta elapsedTime = currentTime - initialTime;
    double timeLeft = 0;

    if (currentStep == 2) { // we compare with the third step
      deltaTime = elapsedTime;
    } else if (currentStep > 2) {
      timeLeft = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     elapsedTime - deltaTime)
                     .count() /
                 1e9 / (currentStep - 2.) * (numSteps - currentStep + 1.);
    }

    if ((currentStep == 0 || currentStep == 2 || currentStep == numSteps - 1) ||
        (currentStep + 1) % reportEvery == 0) {

      int percentage = double(currentStep + 1) / numSteps * 100.;

      std::cout << s << " | ";
      std::cout << std::setw(3) << percentage << "% | ";
      std::cout << std::setw(stepDigits) << currentStep + 1
                << std::setw(stepDigits) << " / " << numSteps;
      if (currentStep > 2 && withTimeEstimate) {
        std::cout << " | remaining: " << std::setw(8) << std::setprecision(2)
                  << std::scientific << timeLeft << std::fixed << " s."
		  << std::endl;
      } else {
        std::cout << std::endl;
      }
    }
    std::cout << std::resetiosflags(std::cout.flags());
  }
}

void LoopPrint::close() {
  if (!mpi->mpiHead())
    return;
  // print timing results
  time_point currentTime;
  currentTime = std::chrono::steady_clock::now();
  std::cout << "Elapsed time: " << std::setprecision(3)
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                currentTime - initialTime)
                .count() / 1e9
            << " s." << std::endl;
}
