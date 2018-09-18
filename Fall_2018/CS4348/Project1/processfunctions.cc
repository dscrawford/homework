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
  for (int i = 1; i < argc; ++i) {
    //Inputreader gives input until
    InputReader file(argv[i]);
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
  }
  
  int adr = 0, status, val, isWrite;
  bool isData = false, isInstruction = false;

  //RAM watching for CPU requesting values
  while ( waitpid(-1, &status, WNOHANG) == 0) {
    read(cpupipe[0], &adr, sizeof(int));
    if (val >= 0) {
      val = ram.read(adr);
      write(rampipe[1], &val, sizeof(int));
      //Check if instruction requires writing
      read(cpupipe[0], &isWrite, sizeof(int));
	if (isWrite == 1) {
	  //Get next input
	  read(cpupipe[0], &adr, sizeof(int));
	  val = ram.read(adr);
	  write(rampipe[1], &adr, sizeof(int));
	  
	  //Read where to store and what to store
	  read(cpupipe[0], &adr, sizeof(int));
	  read(cpupipe[0], &val, sizeof(int));
	  ram.write(adr, val);
	}
    }
  } 
}


void cpuProcess(int* rampipe, int* cpupipe) {
  //Create object class cpu
  CPU cpu(rampipe, cpupipe);
  //run the program
  cpu.runProgram();
}
