#include "elph_plot_app.h"
#include "bandstructure.h"
#include "context.h"
#include "delta_function.h"
#include "el_scattering_matrix.h"
#include "exceptions.h"
#include "io.h"
#include "parser.h"
#include "points.h"

#ifdef HDF5_AVAIL
#include <highfive/H5Easy.hpp>
#endif

void ElPhCouplingPlotApp::run(Context &context) {

  // load ph files
  auto t2 = Parser::parsePhHarmonic(context);
  auto crystal = std::get<0>(t2);
  auto phononH0 = std::get<1>(t2);

  // load electronic files
  auto t1 = Parser::parseElHarmonicWannier(context, &crystal);
  auto crystalEl = std::get<0>(t1);
  auto electronH0 = std::get<1>(t1);

  // load the el-ph coupling
  // Note: this file contains the number of electrons
  // which is needed to understand where to place the fermi level
  auto couplingElPh = InteractionElPhWan::parse(context, crystal, phononH0);

  Eigen::Vector3i mesh;
  if (context.getG2PlotStyle() == "qFixed") {
    mesh = context.getKMesh();
  } else if (context.getG2PlotStyle() == "kFixed") {
    mesh = context.getQMesh();
  } else if (context.getG2PlotStyle() == "allToAll") {
    if (context.getKMesh() != context.getQMesh()) {
      Error("Elph plotting app currently only works with kMesh = qMesh for "
            "allToAll calculations.");
    }
    mesh = context.getKMesh();
  } else {
    Error("Elph plotting app found an incorrect input to couplingPlotStyle.");
  }

  // decide what kind of points path we're going to use
  // ---------------------------

  Points points(crystal);

  if (context.getG2MeshStyle() == "pointsPath") {
    points = Points(crystal, context.getPathExtrema(), context.getDeltaPath());
  } else { //(context.getG2MeshStyle() == "pointsMesh") { // pointsMesh is
           //default
    points = Points(crystal, mesh);
  }

  // loop over points and set up points pairs
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> pointsPairs;

  for (int ik = 0; ik < points.getNumPoints(); ik++) {

    auto thisPoint =
        points.getPointCoordinates(ik, Points::cartesianCoordinates);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> thisPair;

    // create a list of (k,q) pairs, where k is on a path/mesh and q is fixed
    if (context.getG2PlotStyle() == "qFixed") {
      thisPair.first = thisPoint; // set the k point
      thisPair.second = points.crystalToCartesian(
          context.getG2PlotFixedPoint()); // fixed q point
      pointsPairs.push_back(thisPair);

    }
    // create a list of (k,q) pairs, where k is fixed and q is on the path
    else if (context.getG2PlotStyle() == "kFixed") {
      thisPair.first = points.crystalToCartesian(context.getG2PlotFixedPoint());
      thisPair.second = thisPoint; // set the q point
      pointsPairs.push_back(thisPair);
    } else { // if neither point is fixed, it's all to all
      for (int iq = 0; iq < points.getNumPoints(); iq++) {
        thisPair.first = thisPoint;
        thisPair.second =
            points.getPointCoordinates(iq, Points::cartesianCoordinates);
        ;
        pointsPairs.push_back(thisPair);
      }
    }
  }

  if (pointsPairs.size() < size_t(mpi->getSize())) {
    Error(
        "Elph plot app cannot be run with more MPI processes than k,q pairs!");
  }

  // band ranges to calculate the coupling for -----------------------------
  std::pair<int, int> g2PlotEl1Bands = context.getG2PlotEl1Bands();
  std::pair<int, int> g2PlotEl2Bands = context.getG2PlotEl2Bands();
  std::pair<int, int> g2PlotPhBands = context.getG2PlotPhBands();

  // warn users if they have selected a bad band range
  if (g2PlotEl1Bands.second > electronH0.getNumBands()) {
    Warning("The first band range exceeds the possible number of bands. "
            "Setting to max # of bands.");
  }
  if (g2PlotEl2Bands.second > electronH0.getNumBands()) {
    Warning("The second band range exceeds the possible number of bands. "
            "Setting to max # of bands.");
  }
  if (g2PlotPhBands.second > phononH0.getNumBands()) {
    Warning("The phonon band range exceeds the possible number of bands. "
            "Setting to max # of bands.");
  }

  // if not supplied, set first band index is already 0
  // set the max number of bands to the higher end of the ranges
  if (g2PlotEl1Bands.second <= 0 ||
      g2PlotEl1Bands.second >= electronH0.getNumBands()) {
    g2PlotEl1Bands.second = electronH0.getNumBands() - 1;
  }
  if (g2PlotEl2Bands.second <= 0 ||
      g2PlotEl2Bands.second >= electronH0.getNumBands()) {
    g2PlotEl2Bands.second = electronH0.getNumBands() - 1;
  }
  if (g2PlotPhBands.second <= 0 ||
      g2PlotPhBands.second >= phononH0.getNumBands()) {
    g2PlotPhBands.second =
        phononH0.getNumBands() - 1; // minus 1 to account for index from 0
  }

  // check lower bounds
  if (g2PlotPhBands.first < 0 || g2PlotEl1Bands.first < 0 ||
      g2PlotEl2Bands.first < 0) {
    Warning("One of your band range has an index below zero. Setting band "
            "range to index from first band.");
  }

  // check that the first index is minimum of zero
  if (g2PlotEl1Bands.first < 0) {
    g2PlotEl1Bands.first = 0;
  }
  if (g2PlotEl2Bands.first < 0) {
    g2PlotEl2Bands.first = 0;
  }
  if (g2PlotPhBands.first < 0) {
    g2PlotPhBands.first = 0;
  }

  // check if lower band is less than higher band
  if (g2PlotPhBands.first > g2PlotPhBands.second ||
      g2PlotEl1Bands.first > g2PlotEl1Bands.second ||
      g2PlotEl2Bands.first > g2PlotEl2Bands.second) {
    Error("One of your band range index2 - index1 < 0. Check the band ranges.");
  }

  if (mpi->mpiHead()) {
    std::cout << "Coupling to be calculated with (inclusive) band ranges: el1 ["
              << g2PlotEl1Bands.first << " " << g2PlotEl1Bands.second
              << "] el2 [" << g2PlotEl2Bands.first << " "
              << g2PlotEl2Bands.second << "] ph [" << g2PlotPhBands.first << " "
              << g2PlotPhBands.second << "]" << std::endl;
  }

  // Compute the coupling --------------------------------------------------
  std::vector<double> allGs;

  // distribute over k,q pairs
  int numPairs = pointsPairs.size();
  std::vector<size_t> pairParallelIter = mpi->divideWorkIter(numPairs);

  // push back all the state energies
  int numPhBands = g2PlotPhBands.second - g2PlotPhBands.first + 1;
  int numEl1Bands = g2PlotEl1Bands.second - g2PlotEl1Bands.first + 1;
  int numEl2Bands = g2PlotEl2Bands.second - g2PlotEl2Bands.first + 1;
  std::vector<double> energiesK1(numPairs * numEl1Bands);
  std::vector<double> energiesK2(numPairs * numEl2Bands);
  std::vector<double> energiesQ3(numPairs * numPhBands);

  // If mpi pools are used and each process does not have the same
  // amount of work, the code can hang waiting for couplingElph to return.
  // In the worse case, some processes will have one less item.
  // We check if this process's work value is less than the max value across
  // processes, and if so, append a -1 index.
  size_t max = pairParallelIter.size();
  mpi->allReduceMax(&max);
  if (pairParallelIter.size() < max) {
    Eigen::Vector3d kCartesian = Eigen::Vector3d::Zero();
    int numWannier = couplingElPh.getCouplingDimensions()(4);
    Eigen::MatrixXcd eigenVectorK = Eigen::MatrixXcd::Zero(numWannier, 1);
    couplingElPh.cacheElPh(eigenVectorK, kCartesian);
  }

  // we calculate the coupling for each pair, flatten it, and append
  // -------------- it to allGs. Then at the end, we write this chunk to HDF5.

  if (mpi->mpiHead())
    std::cout << "\nCoupling requested for " << numPairs << " k,q pairs."
              << std::endl;

  LoopPrint loopPrint("calculating coupling", "k,q pairs on this process",
                      pairParallelIter.size());
  for (auto iPair : pairParallelIter) {

    loopPrint.update();

    std::pair<Eigen::Vector3d, Eigen::Vector3d> thisPair = pointsPairs[iPair];

    //|g(k,k'=k+q,+q)|^2
    Eigen::Vector3d k1C = thisPair.first;
    Eigen::Vector3d q3C = thisPair.second;
    Eigen::Vector3d k2C = k1C + q3C;

    // need to get the eigenvectors at these three wavevectors
    auto t3 = electronH0.diagonalizeFromCoordinates(k1C);
    auto en1 = std::get<0>(t3);
    Eigen::MatrixXcd eigenVector1 = std::get<1>(t3);

    // second electron eigenvector
    auto t4 = electronH0.diagonalizeFromCoordinates(k2C);
    auto en2 = std::get<0>(t4);
    auto eigenVector2 = std::get<1>(t4);

    std::vector<Eigen::MatrixXcd> eigenVectors2;
    eigenVectors2.push_back(eigenVector2);
    std::vector<Eigen::Vector3d> k2Cs;
    k2Cs.push_back(k2C);

    // phonon eigenvectors
    auto t5 = phononH0.diagonalizeFromCoordinates(q3C);
    auto en3 = std::get<0>(t5);
    auto eigenVector3 = std::get<1>(t5);

    std::vector<Eigen::MatrixXcd> eigenVectors3;
    eigenVectors3.push_back(eigenVector3);
    std::vector<Eigen::Vector3d> q3Cs;
    q3Cs.push_back(q3C);

    // prepare to store energies
    // std::vector<double> tempEn1;
    // std::vector<double> tempEn2;
    // std::vector<double> tempEn3;
    for (int ib1 = g2PlotEl1Bands.first; ib1 <= g2PlotEl1Bands.second; ib1++) {
      // tempEn1.push_back(en1[ib1]*energyRyToEv);
      energiesK1[iPair * numEl1Bands + ib1 - g2PlotEl1Bands.first] =
          en1[ib1] * energyRyToEv;
    }
    for (int ib2 = g2PlotEl2Bands.first; ib2 <= g2PlotEl2Bands.second; ib2++) {
      // tempEn2.push_back(en2[ib2]*energyRyToEv);
      energiesK2[iPair * numEl2Bands + ib2 - g2PlotEl2Bands.first] =
          en2[ib2] * energyRyToEv;
    }
    for (int ib3 = g2PlotPhBands.first; ib3 <= g2PlotPhBands.second; ib3++) {
      // tempEn3.push_back(en3[ib3]*energyRyToEv);
      energiesQ3[iPair * numPhBands + ib3 - g2PlotPhBands.first] =
          en3[ib3] * energyRyToEv;
    }
    // energiesK1.push_back(tempEn1);
    // energiesK2.push_back(tempEn2);
    // energiesQ3.push_back(tempEn3);

    // calculate polar correction
    std::vector<Eigen::VectorXcd> polarData;
    Eigen::VectorXcd polar =
        couplingElPh.polarCorrectionPart1(q3C, eigenVector3);
    polarData.push_back(polar);

    // calculate the elph coupling squared
    couplingElPh.cacheElPh(eigenVector1,
                           k1C); // fourier transform + rotation by k
    couplingElPh.calcCouplingSquared(
        eigenVector1, eigenVectors2, eigenVectors3, q3Cs, k1C,
        polarData); // fourier transform + rotation by k' and q
    auto couplingSq = couplingElPh.getCouplingSquared(
        0); // access the stored matrix elements, which are for the given
            // triplet. Object has bands |g(m,m',nu)|^2

    // the coupling object is coupling at a given set of k,q, for a range of
    // bands band ranges are inclusive of start and finish ones
    for (int ib1 = g2PlotEl1Bands.first; ib1 <= g2PlotEl1Bands.second; ib1++) {
      for (int ib2 = g2PlotEl2Bands.first; ib2 <= g2PlotEl2Bands.second;
           ib2++) {
        for (int ib3 = g2PlotPhBands.first; ib3 <= g2PlotPhBands.second;
             ib3++) {
          allGs.push_back(couplingSq(ib1, ib2, ib3) * energyRyToEv *
                          energyRyToEv);
        }
      }
    }
  } // close pairs loop
  mpi->barrier();
  loopPrint.close();

  mpi->allReduceSum(&energiesK1);
  mpi->allReduceSum(&energiesK2);
  mpi->allReduceSum(&energiesQ3);

  // ==========================================================================
  // now that we've collected all the G values, we want to write them to file.
  if (mpi->mpiHead())
    std::cout << "\nFinished calculating coupling, writing to file."
              << std::endl;

  std::string outFileName = "coupling.elph.phoebe.hdf5";
  std::remove(&outFileName[0]);

  // product of nbands1 * nbands2 * nmodes -- + 1 is because range is inclusive
  size_t bandProd = (g2PlotEl1Bands.second - g2PlotEl1Bands.first + 1) *
                    (g2PlotEl2Bands.second - g2PlotEl2Bands.first + 1) *
                    (g2PlotPhBands.second - g2PlotPhBands.first + 1);

#if defined(HDF5_AVAIL)
  try {
#if defined(MPI_AVAIL) && !defined(HDF5_SERIAL)

    { // need open/close braces so that the HDF5 file goes out of scope

    // open the hdf5 file
    HighFive::FileAccessProps fapl;
    fapl.add(HighFive::MPIOFileAccess{mpi->getComm(), MPI_INFO_NULL});
    HighFive::File file(outFileName, HighFive::File::Overwrite, fapl);

      unsigned int globalSize = numPairs * bandProd;

      // Create the data-space to write g to
      std::vector<size_t> dims(2);
      dims[0] = 1;
      dims[1] = size_t(globalSize);
      HighFive::DataSet dgmat = file.createDataSet<double>(
          "/elphCouplingMat", HighFive::DataSpace(dims));

      // start point and the number of the total number of elements
      // to be written by this process
      size_t start = mpi->divideWorkIter(numPairs)[0] * bandProd;
      size_t offset = start;

      // Note: HDF5 < v1.10.2 cannot write datasets larger than 2 Gbs
      // ( due to max(int 32 bit))/1024^3 = 2Gb overflowing in MPI)
      // In order to be compatible with older versions, we split the tensor
      // into smaller chunks and write them to separate datasets
      // slower, but it will work more often.

      // maxSize represents 2GB worth of std::complex<doubles>, since that's
      // what we write
      auto maxSize = int(pow(1000, 3)) / sizeof(double);
      size_t smallestSize = bandProd; // 1 point pair
      std::vector<int> bunchSizes;

      // determine the size of each bunch of electronic bravais vectors
      // the BunchSizes vector tells us how many are in each set
      int numPairsBunch = mpi->divideWorkIter(numPairs).back() + 1 -
                          mpi->divideWorkIter(numPairs)[0];

      int bunchSize = 0;
      for (int i = 0; i < numPairsBunch; i++) {
        bunchSize++;
        // this bunch is as big as possible, stop adding to it
        if ((bunchSize + 1) * smallestSize > maxSize) {
          bunchSizes.push_back(bunchSize);
          bunchSize = 0;
        }
      }
      // push the last one, no matter the size, to the list of bunch sizes
      bunchSizes.push_back(bunchSize);

      // determine the number of bunches. The last bunch may be smaller than the
      // rest
      int numDatasets = bunchSizes.size();

      // we now loop over these data sets and write each chunk in parallel
      int netOffset = 0; // offset from first bunch in this set to current bunch

      for (int iBunch = 0; iBunch < numDatasets; iBunch++) {

        // we need to determine the start, stop and offset of this
        // sub-slice of the dataset available to this process
        size_t bunchElements = bunchSizes[iBunch] * smallestSize;
        size_t bunchOffset = offset + netOffset;
        netOffset += bunchElements;

        // Each process writes to hdf5
        // The format is ((startRow,startCol),(numRows,numCols)).write(data)
        // Because it's a vector (1 row) all processes write to row=0,
        // col=startPoint with nRows = 1, nCols = number of items this process
        // will write.
        dgmat.select({0, bunchOffset}, {1, bunchElements}).write_raw(&allGs[0]);
      }
    } // end parallel write section
#else
    {

      // throw an error if there are too many elements to write
      unsigned int globalSize = numPairs * bandProd;
      auto maxSize = int(pow(1000, 3)) / sizeof(double);
      if (globalSize > maxSize) {
        Error("Your requested el-ph matrix element file size is greater than "
              "the allowed size\n"
              "for a single write by HDF5. Either compile Phoebe with a "
              "parallel copy of HDF5 or\n"
              "request to output fewer matrix elements. ");
      }

      // call an mpi collective to grab allGs
      std::vector<double> collectedGs;
      if (mpi->mpiHead())
        collectedGs.resize(globalSize);
      mpi->allGatherv(&collectedGs, &allGs);

      // write elph matrix elements
      HighFive::File file(outFileName, HighFive::File::Overwrite);
      file.createDataSet("/elphCouplingMat", collectedGs);
    }
#endif
    // now we write a few other pieces of smaller information using only mpiHead

    if (mpi->mpiHead()) {

      HighFive::File file(outFileName, HighFive::File::ReadWrite);

      // shape the points pairs list into a format that can be written
      Eigen::MatrixXd pointsTemp(pointsPairs.size(), 6);
      Eigen::MatrixXd pointsTempCart(pointsPairs.size(), 6);

      for (size_t iPair = 0; iPair < pointsPairs.size(); iPair++) {

        auto thisPair = pointsPairs[iPair];
        Eigen::Vector3d k1C = points.cartesianToCrystal(thisPair.first);
        Eigen::Vector3d q3C = points.cartesianToCrystal(thisPair.second);
        for (int i : {0, 1, 2}) {
          pointsTemp(iPair, i) = k1C(i);
          pointsTemp(iPair, i + 3) = q3C(i);
          pointsTempCart(iPair, i) = thisPair.first(i);
          pointsTempCart(iPair, i + 3) = thisPair.second(i);
        }
      }
      // write the points pairs to file
      file.createDataSet("/pointsPairsCrystal", pointsTemp);
      file.createDataSet("/pointsPairsCartesian", pointsTempCart);

      // write the band ranges
      std::vector<int> temp;
      temp.push_back(g2PlotEl1Bands.first);
      temp.push_back(g2PlotEl1Bands.second);
      file.createDataSet("/elBandRange1", temp);
      temp.clear();
      temp.push_back(g2PlotEl2Bands.first);
      temp.push_back(g2PlotEl2Bands.second);
      file.createDataSet("/elBandRange2", temp);
      temp.clear();
      temp.push_back(g2PlotPhBands.first);
      temp.push_back(g2PlotPhBands.second);
      file.createDataSet("/phModeRange", temp);

      // write energies to file
      file.createDataSet("/elEnergies1", energiesK1);
      file.createDataSet("/elEnergies2", energiesK2);
      file.createDataSet("/phEnergies", energiesQ3);
      std::string energyUnit = "eV";
      file.createDataSet("/energyUnit", energyUnit);
    }
  } catch (std::exception &error) {
    Error("Issue writing el-ph Wannier representation to hdf5.");
  }
// close else for HDF5_AVAIL
#else
  Error("You cannot output the elph matrix elements to HDF5 because your copy "
        "of \n"
        "Phoebe has not been compiled with HDF5 support.");
#endif
}

// TODO is there an issue where half the poitns are in crystal and half in
// cartesian
// TODO why do cartesian coords go past 1?
// TODO fix checkRequirements
// TODO write tutorial
// TODO write tests
// TODO check with kfixed, qfixed, none fixed + path vs mesh
// TODO check with and without HDF5, as well as with HDF5_SERIAL

void ElPhCouplingPlotApp::checkRequirements(Context &context) {
  throwErrorIfUnset(context.getElectronH0Name(), "electronH0Name");
  throwErrorIfUnset(context.getPhFC2FileName(), "phFC2FileName");
  if (context.getG2PlotStyle() == "pointsPath") {
    throwErrorIfUnset(context.getPathExtrema(), "points path extrema");
  }
  if (context.getG2PlotStyle() == "pointsMesh" &&
      context.getG2PlotStyle() == "qFixed") {
    throwErrorIfUnset(context.getKMesh(), "kMesh");
  }
  if (context.getG2PlotStyle() == "pointsMesh" &&
      context.getG2PlotStyle() == "kFixed") {
    throwErrorIfUnset(context.getKMesh(), "qMesh");
  }
  throwErrorIfUnset(context.getElphFileName(), "elphFileName");
}
