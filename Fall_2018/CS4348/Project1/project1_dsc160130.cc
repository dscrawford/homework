#include "project1_dsc160130.h"

int main(int argc, char** argv) {
  //Only execute if two 
  if (argc < 2) {
    std::cout << "ERROR: Need to supply arguments after executable "
	      << "(EX: ./program1_dsc160130 sample.txt sample2.txt)" << std::endl;
    exit(EXIT_FAILURE);
  }

  //in rampipe ram writes, in cpupipe cpupipe writes
  int rampipe[2];
  int cpupipe[2];
  //Attempt to pipe both
  if (pipe(rampipe) && pipe(cpupipe)) {
    std::cout << "ERROR: Failed to pipe" << std::endl;
    return 1;
  }
  
  pid_t pid = fork();
  if (pid == 0) {
    //child process
    ramProcess(argc, argv, rampipe, cpupipe);
  }

  else if (pid > 0) {
    //parent process
    cpuProcess(rampipe, cpupipe);
  }

  else {
    //fork failed
    std::cout << "fork failed" << std::endl;
    return 1;
  }
  
  return 0;
}
