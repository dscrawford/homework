#include "project1_dsc160130.h"
void ramProcess(int argc, char** argv, int* rampipe, int* cpupipe) {
  RAM ram;
  
  //instruction count
  int instrc = 0;

  //Read every file in argv and then read all of those files
  //Start at arguments after the executable
  for (int i = 1; i < argc; ++i) {
    //Read InputReader until no files are left
    InputReader file(argv[i]);
    while (!file.eof()) {
      std::string str;
      if ( (str = file.read()) != "NULL" ) {
	//Write to ram.
	ram.write(instrc, stoi(str));
	instrc++;
      }
    }
  }

  write(rampipe[1], &instrc, sizeof(int));
  //Send all values in memory to the cpu
    for (int i = 0; i < instrc; ++i) {
    int temp = ram.read(i);
    write(rampipe[1], &temp, sizeof(int));
  }
  
  int isRunning = 0;
  while ()
    read(cpupipe[0], &isRunning, sizeof(int));
  std::cout << "I just read this" << std::endl;

  std::cout << "isRunning: " << isRunning << std::endl;
}

void cpuProcess(int* rampipe, int* cpupipe) {
  CPU cpu;

  //Get instruction count
  int instrc, isRunning;
  read(rampipe[0], &instrc, sizeof(int));
  std::cout << "cpu count: " << instrc << std::endl;
  
  cpu.getInstructs(rampipe, instrc);

  isRunning = 0;
  write(cpupipe[1], &isRunning, sizeof(int));
  std::cout << "I just wrote this" << std::endl;
  std::cout << isRunning << std::endl;
}

