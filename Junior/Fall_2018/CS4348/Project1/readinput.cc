#include "readinput.h"

//Initialize with the filename, try to open the file.
InputReader::InputReader(std::string FILENAME) {
  file.open(FILENAME, std::ios::in);
  if (!file) {
    std::cerr << "ERROR: unable to open input file" << std::endl;
    _exit(1);
  }
}

//Read the next line in the input. Loop will be implemented externally.
std::string InputReader::read() {
  std::string line;
  std::stringstream  ss;

  //Loop will search for a digit, if it finds nothing it will iterate again.
  do {
    std::getline(file, line);

    unsigned int i = 0;
    //Read if it starts with a digit
    if (line[0] ==  '.') {
      ss << line[0];
      i++;
    }
    //Read through string to get int, only read integers
    for ( ; i < line.size() && isdigit(line[i]) ; ++i) {
      ss << line[i];
    } 
  } while ( ss.str() == "" && !eof());

  //Return integer or the string NULL to say nothing else was found
  if ( ss.str() != "" )
    return ss.str();
  else
    return "NULL";
}

bool InputReader::eof() {
  return file.eof();
}
