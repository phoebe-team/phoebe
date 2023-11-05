#include "utilities.h"
#include <fstream>
#include <iostream>
#include <mpiHelper.h>
#include <unistd.h>
#include <iomanip>
#include <regex>

int mod(const int &a, const int &b) { return (a % b + b) % b; }

int compress3Indices(const int &i1, const int &i2, const int &i3,
                     const int &size1, const int &size2, const int &size3) {
  (void)size1;
  return i1 * size2 * size3 + i2 * size3 + i3;
}

std::tuple<int, int, int> decompress3Indices(const int &iTot, const int &size1,
                                             const int &size2,
                                             const int &size3) {
  (void)size1;
  int i1 = iTot / (size2 * size3);
  int remainder = iTot - i1 * size2 * size3;
  int i2 = remainder / size3;
  remainder -= i2 * size3;
  int i3 = remainder;
  return std::make_tuple(i1, i2, i3);
}

int compress2Indices(const int &i1, const int &i2, const int &size1,
                     const int &size2) {
  (void)size1;
  return i1 * size2 + i2;
}

std::tuple<int, int> decompress2Indices(const int &iTot, const int &size1,
                                        const int &size2) {
  (void)size1;
  int i1 = iTot / size2;
  int i2 = iTot - i1 * size2;
  return std::make_tuple(i1, i2);
}

std::tuple<double, double> memoryUsage() {

  // based on: https://www.tutorialspoint.com/how-to-get-memory-usage-under-linux-in-cplusplus
  // see https://man7.org/linux/man-pages/man5/proc.5.html
  double vm_usage = 0.;
  double resident_set = 0.;
  std::ifstream stat_stream("/proc/self/stat",
                            std::ios_base::in); // get info from proc directory
  // create some variables to get info
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string uTime, sTime, cuTime, csTime, priority, nice;
  std::string O, itRealValue, startTime;
  unsigned long vSize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
      tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> uTime >>
      sTime >> cuTime >> csTime >> priority >> nice >> O >> itRealValue >>
      startTime >> vSize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024; // for x86-64 is configured to use 2MB pages
  vm_usage = double(vSize) / 1024.0;
  resident_set = double(rss) * double(page_size_kb);

  mpi->allReduceSum(&vm_usage);
  mpi->allReduceSum(&resident_set);
  if ( mpi->mpiHead() ) {
    std::cout << std::setprecision(4);
    std::cout << "Snapshot of Phoebe's memory usage:\n";
    std::cout << "VM: " << vm_usage/1024./1024. << " (GB). RSS: "
              << resident_set/1024./1024. << " (GB)\n" << std::endl;
  }

  return std::make_tuple(vm_usage, resident_set);
}

double findMaxRelativeDifference(const Eigen::Tensor<double,3> &x,
                                 const Eigen::Tensor<double,3> &xRef) {
  if (!(x.dimensions() == xRef.dimensions())) {
    Error("Can't compare inconsistent tensors");
  }
  Eigen::Tensor<double, 3> diff = ((x - xRef) / xRef).abs();
  double maxDiff = 1.e20;
  for (int i = 0; i < diff.dimension(0); i++) {
    for (int j = 0; j < diff.dimension(1); j++) {
      for (int k = 0; k < diff.dimension(2); k++) {
        if (diff(i, j, k) < maxDiff) {
          maxDiff = diff(i, j, k);
        }
      }
    }
  }
  if (maxDiff == 1.e20) {
    Error("wrong bounds in findMaxRelativeDifference");
  }
  return maxDiff;
}

// helper to break up strings by comma and spaces and quote marks
std::vector<std::string> tokenize(const std::string str) {

  // break them to remove commas and spaces
  const std::regex re(R"([\s|,|\"]+)");
  std::sregex_token_iterator it{ str.begin(), str.end(), re, -1 };
  std::vector<std::string> tokenized{ it, {} };

  // remove empty strings
  tokenized.erase(std::remove_if(tokenized.begin(), tokenized.end(),
                     [](std::string const& s) {
                         return s.size() == 0;
                     }), tokenized.end());
  return tokenized;
}

