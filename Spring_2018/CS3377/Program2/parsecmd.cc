//Made by Daniel Crawford(dsc160130@utdallas.edu) on 1/28/2018
//CS 3377.501
#include "program2.h"

void parsecmd(int argc, char** argv) {
  try {
    //Create command line parser and option menu
    TCLAP::CmdLine cmd("Insert a inputfile and program will output the file and edit it",' ', "2.0");
    //Find argument for outputfiles
    TCLAP::ValueArg<std::string> outfileArg("o","outfile","The name of the output file",false,
					    "outfile.txt", "output filename");
    //Find argument for input files
    TCLAP::UnlabeledValueArg<std::string> infileArg("input","Input file",true,"null","input filename", true);
    //Find arguments for upper/lower case switches
    TCLAP::SwitchArg lowerCaseSwitch("l","lower","Convert all text to lowercase",cmd, false);    
    TCLAP::SwitchArg upperCaseSwitch("u","upper","Convert all text to uppercase",cmd, false);
    //Add arguments to command line parser
    cmd.add(outfileArg);
    cmd.add(infileArg);
    cmd.parse(argc, argv);
    
    //All arguments will be stored in a map
    std::map<int, std::string> parseResults;

    //While storing results in map, it will check to see if all flags are valid.
    //All commands done by TCLAP will now be accessible through std::map
    putParseResultsInMap(cmd, outfileArg, infileArg, upperCaseSwitch.getValue(), lowerCaseSwitch.getValue());
  }
  catch (TCLAP::ArgException &e) {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }
}

void putParseResultsInMap(TCLAP::CmdLine &cmd, TCLAP::ValueArg<std::string> &outfileArg, TCLAP::UnlabeledValueArg<std::string> &infileArg, bool uppercase, bool lowercase) {
  std::list<TCLAP::Arg *> list = cmd.getArgList();
  std::list<TCLAP::Arg *>::const_iterator itr;
  //Iterate through whole argument list, don't have to account for arguments help and version
  //Will assign each argument in the map
  for (itr = list.begin(); itr != list.end(); ++itr) {
    std::string str = (*itr)->getName();
    if (str == "upper" && uppercase)
      cmdmap[filetouppercase] = "true";
    else if (str == "lower" && lowercase)
      cmdmap[filetolowercase] = "true";
    else if (str == "outfile")
      cmdmap[outputfilename] = outfileArg.getValue();
    else if (str == "input")
      cmdmap[inputfilename] = infileArg.getValue();    
  }
}
