#include "project1_dsc160130.h"

int main(int argc, char** argv) {
  //Only execute if two or three arguments. If only 2, default to 10 for timer
  if (argc != 2 && argc != 3) {
    std::cout << "ERROR: Invalid arguments, need input file then timer"
	      << "(EX: ./program1_dsc160130 sample.txt 10)" << std::endl;
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
    cpuProcess(rampipe, cpupipe);
    return 0;
  }
  else if (pid == 0) {
    //child process
    ramProcess(argc, argv, pid, rampipe, cpupipe);
    return 0;
  }
  else {
    //fork failed
    std::cout << "fork failed" << std::endl;
    return 1;
  }
  return 0;
}
