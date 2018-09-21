#include "project1_dsc160130.h"
void ramProcess(int argc, char** argv, pid_t pid, int* rampipe, int* cpupipe) {
  //Create object class ram
  RAM ram;
  //instruction count


  //Close unused pipes
  close(rampipe[0]);
  close(cpupipe[1]);

  //Read every file in argv and then read all of those files
  //Start at arguments after the executable
  int address = 0;
  //Inputreader gives input until

  InputReader file(argv[1]);
  while (!file.eof()) {
    std::string str;
    if ( (str = file.read()) != "NULL" ) {
      //Write where data starts
      if (str[0] == '.') { //If string starts with period
	str.erase(0,1); //Erase period
	address = stoi(str); //Set new address	
      }
      //Write instruction
      else {
	ram.write(address, stoi(str));
	address++;
      }
    }
  }
  /*int c = 5;
  for (int i = 0; i < 2000; ++i) {
    std::cout << "arr[" << i << "]: " << ram.read(i) << ", ";
    if (c == 0) {
      std::cout << std::endl;
      c = 5;
    }
    c--;
    }
  */
  int timer = stoi(std::string(argv[2]));
  write(rampipe[1], &timer, sizeof(int));

  

  //Get the number of instructions to run before an interrupt

  //Write to CPU
  int adr = 0, status, val, isWrite;

  //RAM watching for CPU requesting values
  while ( waitpid(-1, &status, WNOHANG) == 0) {
    read(cpupipe[0], &adr, sizeof(int));
    
    val = ram.read(adr);
    write(rampipe[1], &val, sizeof(int));
    //Check if instruction requires writing
    read(cpupipe[0], &isWrite, sizeof(int));
    if (isWrite == 1) {
      //Get next input
      int adr2;
      read(cpupipe[0], &adr2, sizeof(int));
      if (adr2 >= 0) {
	val = ram.read(adr2);
	std::cout << "GOT VAL " << val << std::endl;
	write(rampipe[1], &val, sizeof(int));
      }
      //Read where to store and what to store
      read(cpupipe[0], &adr, sizeof(int));
      std::cout << "WRITING ADDRESS " << adr << std::endl;
      read(cpupipe[0], &val, sizeof(int));
      std::cout << "WRITING VAL " << val << std::endl;
      ram.write(adr, val);
    }
  }
}


void cpuProcess(int* rampipe, int* cpupipe) {
  //Create object class cpu
  CPU cpu(rampipe, cpupipe);
  //run the program
  cpu.runProgram();
}
