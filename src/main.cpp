#include "app.h"
#include "context.h"
#include "io.h"
#include "mpi/mpiHelper.h"

int main(int argc, char **argv) {

  // here launch parallel environment
  // Call proxy function from MPI Helper, which makes mpi object
  // globally available.
  initMPI();

  // setup input/output
  IO io(argc, argv);
  io.welcome();

  // Print parallelization info
  if (mpi->mpiHead()) parallelInfo();

  // Read user input file
  Context context; // instantiate class container of the user input
  context.setupFromInput(io.getInputFileName()); // read the user input

  // decide which app to use
  std::unique_ptr<App> app = App::loadApp(context.getAppName());

  // check that the user passed all the necessary input
  app->checkRequirements(context);

  // launch it
  app->run(context);
  // exiting program
  io.goodbye();

  // here close parallel environment
  // make sure all processes finish before printing end info
  mpi->finalize();

  return (0);
}
