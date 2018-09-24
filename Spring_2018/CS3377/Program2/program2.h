//Made by Daniel Crawford(dsc160130@utdallas.edu) on 1/28/2018
//CS 3377.501
#include <fstream>
#include <iostream>
#include <tclap/CmdLine.h>
#include <tclap/Arg.h>
#include <map>
#include <list>
#include <string>

void openFiles(std::ofstream &, std::ifstream &);
void editFile();
void putParseResultsInMap(TCLAP::CmdLine &, TCLAP::ValueArg<std::string> &,
			  TCLAP::UnlabeledValueArg<std::string> &, bool, bool);
void parsecmd(int argc, char** argv);

enum mapIndices {outputfilename, inputfilename,filetolowercase, filetouppercase};

extern std::map<int, std::string> cmdmap;
