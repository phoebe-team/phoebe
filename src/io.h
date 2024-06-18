#ifndef IO_H
#define IO_H

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "context.h"

/** class used to parse Phoebe command line arguments, and to redirect output
 * from std::cout to output-File
 */
class IO {
private:
  std::ofstream outputFile;
  std::string outputFileName;
  std::string inputFileName;

public:
  /** Constructor: parses Phoebe's command line arguments and set up output
   */
  IO(int argc, char *argv[]);

  /** Destructor: closes the output file
   */
  ~IO();

  /** Returns the name of the input file parsed from the command line
   * Used by Context to read the input file.
   */
  std::string getInputFileName();

  /** Prints the banner
   */
  static void welcome();

  /** Prints a closing message
   */
  static void goodbye(Context &context);
};

/** Class used to time loops, and provide the user with a report on a loop
 * and how it is proceeding, estimating also how much time is left.
 */
class LoopPrint {
private:
public:
  /** Constructor.
   * It also constructs a reference time 0 to measure elapsed time.
   * @param task: a name describing what's done in this loop.
   * @param step: the name describing what is incremented in the loop
   * @param numSteps: size of the loop.
   * These parameters will be inserted in the string:
   * "Started {task} with {numSteps} {step}." e.g.
   * "Started {q-point loop} with {100} {q-points}.".
   */
  LoopPrint(const std::string &task, const std::string &step,
            const size_t &numSteps);

  /** Method to update on the progress of the loop
   * It must be called in the loop numSteps times.
   * If it's called a different number of times, the screen report will be
   * made incorrectly.
   */
  void update(const bool &withTimeEstimate=true);

  /** Close loopInfo and print summary of loop execution time.
   */
  void close();

private:
  typedef std::chrono::steady_clock::time_point time_point;
  typedef std::chrono::steady_clock::duration time_delta;
  int reportEvery;
  size_t numSteps;
  std::string task;
  std::string step;
  size_t currentStep = -1;
  time_point initialTime;
  time_delta deltaTime;
  int stepDigits;
};

#endif
