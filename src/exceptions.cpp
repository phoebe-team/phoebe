#include "exceptions.h"
#include "mpiHelper.h"
#include <iostream>
#include <string>

Error::Error(const std::string &errMessage, const int &errCode) {
  if (errCode != 0) {
    if (mpi->mpiHead()) {
      std::cout << "\nError!" << std::endl;
      std::cout << errMessage << "\n" << std::endl;
    }
    mpi->barrier();
    mpi->finalize();
    exit(errCode);
  }
}

DeveloperError::DeveloperError(const std::string &errMessage, const int &errCode) :
  Error("Developer Error: " + errMessage, errCode) {
}

Warning::Warning(const std::string &errMessage) {
  if (!mpi->mpiHead())
    return;
  std::cout << "\nWARNING: " << errMessage << "\n" << std::endl;
}
