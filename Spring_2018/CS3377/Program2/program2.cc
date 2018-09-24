//Made by Daniel Crawford(dsc160130@utdallas.edu) on 1/28/2018
//CS 3377.501
#include "program2.h"
std::map<int,std::string> cmdmap;

int main(int argc, char** argv) {
  parsecmd(argc, argv);
  editFile();
  return 0;
}
