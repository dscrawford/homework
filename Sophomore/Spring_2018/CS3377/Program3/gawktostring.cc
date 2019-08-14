//Made by Daniel Crawford on 2/19/2018(dsc160130)
//Section 3377.501
#include "program3.h"
#include <stdio.h>
std::string getgawkcommand(std::string ARGS) {
  //Issue gawk command for version
  std::string command = PATH + " " + ARGS;
  FILE *stream = popen(command.c_str(), "r");

  //If the attempt to access the command fails.
  if (!stream) {
    std::cerr << "error: piping/forking unsuccessful" << std::endl;
    exit(EXIT_FAILURE);
  }

  //Place all contents of awk request into std::string
  char c;
  std::string str;
  while ( (c = fgetc(stream)) != EOF) {
    str = str + c;
  }

  //If string is empty, then inform the user that they
  if (str == "") {
    std::cerr << "error: no ouput was returned by gawk" << std::endl;
    exit(EXIT_FAILURE);
  }
  //Close stream, if it failed then notify user
  if (pclose(stream) == -1) {
    std::cerr << "error: an error has occured while attempting to close stream" << std::endl;
    exit(EXIT_FAILURE);
  }
  return str;
}
