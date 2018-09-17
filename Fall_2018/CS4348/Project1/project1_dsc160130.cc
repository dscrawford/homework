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
  if (pipe(rampipe)) {
    std::cerr << "ERROR: Failed to pipe rampipe" << std::endl;
    return 1;
  }
  if (pipe(cpupipe)) {
    std::cerr << "ERROR: Failed to pipe cpupipe" << std::endl;
    return 1;
  };
  
  pid_t pid = fork();
  if (pid > 0) {
    //parent process
    ramProcess(argc, argv, pid, rampipe, cpupipe);
    return 0;
  }
  else if (pid == 0) {
    //child process
    cpuProcess(rampipe, cpupipe);
    return 0;
  }
  else {
    //fork failed
    std::cout << "fork failed" << std::endl;
    return 1;
  }
  
  return 0;
}
