#ifndef READINPUT_H
#define READINPUT_H

#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <unistd.h>
#include <iostream>

class InputReader { 
 private:
  std::fstream file;
 public:
  InputReader(std::string); //Iniliazes file
  std::string read(); //returns string of integer, "NULL" if none were found
  bool eof();
};

#endif // READINPUT_H
