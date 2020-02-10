#include <iostream>
#include <string>
#include <fstream>
#include "bigram.h"

using namespace std;

bool checkArgs(int argc, char** args) {
  if (argc != 4) {
    cerr << "Error: Insufficient amount of arguments" << endl;
    return false;
  }

  bool valid = true;
  for (int i = 1; i < 3; ++i) {
    ifstream ifile(args[i]);
    if (!ifile) {
      cerr << "Error: Unable to find file \"" << args[i] << "\"" << endl;
      valid = false;
    }
  }
  
  if (*args[3] != '0' && *args[3] != '1') {
    cerr << "Error: Argument \"" << args[3] << "\" must be either 1 or 0" << endl;
    valid = false;
  }

  return valid;
}

int main(int argc, char** args) {
  if (!checkArgs(argc, args))
    return 1;
  Bigram(std::string(args[1]));
  return 0;
}

