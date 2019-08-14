#include "project1_dsc160130.h"
void ramProcess(int argc, char** argv, pid_t pid, int* rampipe, int* cpupipe) {
  //Create object class ram
  RAM ram;
  //Close unused pipes
  close(rampipe[0]);
  close(cpupipe[1]);

  //Read every file in argv and then read all of those files
  //Start at arguments after the executable
  int address = 0;
  //Inputreader gives input until

  //Read the file and initialize ram
  InputReader file(argv[1]);
  while (!file.eof()) {
    std::string str;
    if ( (str = file.read()) != "NULL" ) {
      //Load new address.
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

  int timer;
  //Get the number of instructions to run before an interrupt, if none provided
  //default to 10
  if (argc == 3)
    timer = stoi(std::string(argv[2]));
  else
    timer = 10;
  write(rampipe[1], &timer, sizeof(int));

  //Write to CPU
  int adr = 0, val, type;

  //RAM watching for CPU requesting reads and writes.
  while (true) {
    //Get address of where to read from CPU
    read(cpupipe[0], &adr, sizeof(int));

    //Fetch instruction
    val = ram.read(adr);
    write(rampipe[1], &val, sizeof(int));

    //Check if instruction requires writing, otherwise next iteration.
    read(cpupipe[0], &type, sizeof(int));
    //Write type
    if (type == 1) {
      //Get next line possibly, if negative skip
      int adr2;
      read(cpupipe[0], &adr2, sizeof(int));
      if (adr2 >= 0) {
	val = ram.read(adr2);
	write(rampipe[1], &val, sizeof(int));
      }
      //Read where to store and what to store
      read(cpupipe[0], &adr, sizeof(int));
      read(cpupipe[0], &val, sizeof(int));
      //Write to memory.
      ram.write(adr, val);
    }
    //Exit type
    if (type == 2) {
      break;
    }
  }
}


void cpuProcess(int* rampipe, int* cpupipe) {
  //Create object class cpu
  CPU cpu(rampipe, cpupipe);
  //run the program
  cpu.runProgram();
}
