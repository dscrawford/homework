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
  
  int adr = 0, status, val;
  bool isData = false;

  //RAM watching for CPU requesting values
  while ( waitpid(-1, &status, WNOHANG) == 0) {
    std::cout << "Checkpoint0" << std::endl;
    read(cpupipe[0], &adr, sizeof(int));
    std::cout << "Checkpoint1" << std::endl;
    if (adr >= 0) {
      val = ram.read(adr);
      write(rampipe[1], &val, sizeof(int));
    }

    if (!isData) {
      std::cout << "Checkpoint2" << std::endl;
      //Skip this if next input is Data
      switch(val) {
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 9:
      case 20:
      case 21:
      case 22:
	isData = true;
	break;
      case 7:
      case 23:
      case 24:
      case 27:
      case 28:
	//Read next line of input
	read(cpupipe[0], &adr, sizeof(int));
	val = ram.read(adr);
	write(rampipe[1], &val, sizeof(int));
	
	//Read value and address to write it in
	read(cpupipe[0], &adr, sizeof(int));
	read(cpupipe[0], &val, sizeof(int));
	ram.write(adr, val);
	break;
      case 50:
	_exit(0);
	break;
      }
    }
    else {
      std::cout << "Checkpoint3" << std::endl;
      isData = false;
    }
    
  }
}

void cpuProcess(int* rampipe, int* cpupipe) {
  //Create object class cpu
  CPU cpu(rampipe, cpupipe);
  //run the program
  cpu.runProgram();
}
