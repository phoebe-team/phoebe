#include "parser.h"

std::tuple<Crystal, PhononH0> Parser::parsePhHarmonic(Context &context) {

  std::string fc2FileName = context.getPhFC2FileName();

  // try to use file extension to determine which file we have

  // from qe we expect a seedname.fc file
  if(fc2FileName.find(".fc") != std::string::npos) {
    if(mpi->mpiHead()) std::cout << "Parsing FC2s from QE." << std::endl;
    return QEParser::parsePhHarmonic(context);
  }
  // the jdftx filename is seedname.phononHsub
  else if(fc2FileName.find("phononOmegaSq") != std::string::npos) {
    if(mpi->mpiHead()) std::cout << "Parsing FC2s from JDFTx." << std::endl;
    return JDFTxParser::parsePhHarmonic(context);
  }
  // otherwise try to parse as a QE 
  else if (!context.getPhonopyDispFileName().empty()) {
    if(mpi->mpiHead()) std::cout << "Parsing FC2s from Phonopy." << std::endl;
    return PhonopyParser::parsePhHarmonic(context);
  }
  else {
    Error("Unrecognised FC2 format!\n"
    "-- to use QE, supply a file with a .fc extension\n"
    "-- to use JDFTx, supply a file with the .phononOmegaSq extension\n"
    "-- to use phonopy, set the phonopyDispFileName input variable and supply a .hdf5 file\n");
  }
}

std::tuple<Crystal, ElectronH0Fourier> Parser::parseElHarmonicFourier(
        Context &context) {
  return QEParser::parseElHarmonicFourier(context);
}

std::tuple<Crystal, ElectronH0Wannier> Parser::parseElHarmonicWannier(
            Context &context, Crystal *inCrystal) {
  return QEParser::parseElHarmonicWannier(context, inCrystal);
}
