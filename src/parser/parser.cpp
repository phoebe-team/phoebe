#include "parser.h"

std::tuple<Crystal, PhononH0> Parser::parsePhHarmonic(Context &context) {

  std::string fc2FileName = context.getPhFC2FileName();

  // try to use file extension to determine which file we have

  // from qe we expect a seedname.fc file
  if(fc2FileName.find(".fc") != std::string::npos) {
    if(mpi->mpiHead()) std::cout << "Parsing FC2s from QE." << std::endl;
    return QEParser::parsePhHarmonic(context);
  }
  // TODO we should simplify this parsing, it's very confusing to trigger the right 
  // parser if one wants to use JDFTx elph and phonopy FC3
  else if(!context.getJDFTxScfOutFile().empty() && !context.getPhonopyDispFileName().empty()) {
    if(fc2FileName.find("jdftx") != std::string::npos) {
      if(mpi->mpiHead()) std::cout << "Parsing FC2s from JDFTx." << std::endl;
        return JDFTxParser::parsePhHarmonic(context);
    } else {
      if(mpi->mpiHead()) std::cout << "Parsing FC2s from Phonopy." << std::endl;
        return PhonopyParser::parsePhHarmonic(context);
    }
  }
  // the jdftx filename should point to totalE.out
  else if(!context.getJDFTxScfOutFile().empty()) {
    if(mpi->mpiHead()) std::cout << "Parsing FC2s from JDFTx." << std::endl;
    return JDFTxParser::parsePhHarmonic(context);
  }
  // otherwise try to parse as phonopy format 
  else if (!context.getPhonopyDispFileName().empty()) {
    if(mpi->mpiHead()) std::cout << "Parsing FC2s from Phonopy." << std::endl;
    return PhonopyParser::parsePhHarmonic(context);
  }
  else {
    Error("Unrecognised FC2 format!\n"
    "-- to use QE, supply a file with a .fc extension\n"
    "-- to use JDFTx, supply a jdftx.elph.phoebe.hdf5 file and set the JDFTxScfOutFile variable to the path to totalE.out\n"
    "-- to use phonopy, set the phonopyDispFileName input variable and supply a .hdf5 file\n");
    return QEParser::parsePhHarmonic(context); // this isn't ever reached, just silences warnings
  }
}

std::tuple<Crystal, ElectronH0Fourier> Parser::parseElHarmonicFourier(
        Context &context) {
  return QEParser::parseElHarmonicFourier(context);
}

std::tuple<Crystal, ElectronH0Wannier> Parser::parseElHarmonicWannier(
            Context &context, Crystal *inCrystal) {

  // try to use set file to determine which file we have

  // from qe we expect a seedname. file
  if(!context.getWannier90Prefix().empty() || context.getElectronH0Name().find("_tb.dat") != std::string::npos) {
    if(mpi->mpiHead()) std::cout << "\nParsing Wannier Hamiltonian from QE." << std::endl;
    return QEParser::parseElHarmonicWannier(context, inCrystal);
  }
  // the jdftx filename should point to totalE.out
  else if(!context.getJDFTxScfOutFile().empty()) {
    if(mpi->mpiHead()) std::cout << "\nParsing Wannier Hamiltonian from JDFTx." << std::endl;
    return JDFTxParser::parseElHarmonicWannier(context);
  } else {
    Error("Unrecognised Wannier Hamiltonian format!\n"
    "-- to use Wannier90, supply a file with a *_tb.dat extension\n"
    "-- to use JDFTx, supply a jdftx.elph.phoebe.hdf5 file and set the JDFTxScfOutFile variable to the path to totalE.out\n");
    return QEParser::parseElHarmonicWannier(context,inCrystal); // this isn't ever reached, just silences warnings
  }
}
