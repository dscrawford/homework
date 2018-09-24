//Made by Daniel Crawford(dsc160130@utdallas.edu) on 1/28/2018
//Cs 3377.501
#include "program2.h"

//This contains the operations for copying the input file to the outputfile, uses the map to determine the arguments
void editFile() {
    //If both arguments are true, it will send an error to standard error and exit the program.
    if (cmdmap[filetolowercase] == "true" && cmdmap[filetouppercase] == "true") {
      std::cerr << "error: Cannot make the file both all uppercase and all lowercase" << std::endl;
      exit(EXIT_FAILURE);
    }
    //Otherwise it will try to open the output files and then modify them.
    else {
      std::ifstream inputfile;
      std::ofstream outputfile;
      openFiles(outputfile, inputfile);
      
      std::string line;
      std::locale loc;

      //Copy the contents of the file
      while(!inputfile.eof()) {
	getline(inputfile, line);
	for (int i = 0; i < line.length(); ++i) {	 
	  if (cmdmap[filetolowercase] == "true")
 	    outputfile << std::tolower(line[i], loc);	  
	  else if (cmdmap[filetouppercase] == "true")
	    outputfile << std::toupper(line[i], loc);
	  else
	    outputfile << line[i];
	}
	outputfile << std::endl;
      }
    }
}

//Simply open the files designated in the map, if it doesn't work, it will warn the user and exit the program.
void openFiles(std::ofstream &outputfile, std::ifstream &inputfile) {
    //Attempt to open the output file
    std::string OUTPUT_FILE_NAME = cmdmap[outputfilename];
    outputfile.open(OUTPUT_FILE_NAME.c_str(), std::fstream::out);
    if (!outputfile) {
      std::cerr << "error: Was not able to open output file " << OUTPUT_FILE_NAME << std::endl;
      exit(EXIT_FAILURE);
    }

    //Attempt to open the input file
    std::string INPUT_FILE_NAME = cmdmap[inputfilename];
    inputfile.open(INPUT_FILE_NAME.c_str(), std::fstream::in);
    if (!inputfile) {
      std::cerr << "error: Was not able to open input file " << INPUT_FILE_NAME << std::endl;
      exit(EXIT_FAILURE);
    }
}


