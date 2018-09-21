#include "readinput.h"

InputReader::InputReader(std::string FILENAME) {
  file.open(FILENAME, std::ios::in);
  if (!file) {
    std::cerr << "ERROR: unable to open input file" << std::endl;
    _exit(1);
  }
}

std::string InputReader::read() {
  std::string line;
  std::stringstream  ss;

  do {
    std::getline(file, line);

    unsigned int i = 0;
    //Read if it starts with a digit
    if (line[0] ==  '.') {
      ss << line[0];
      i++;
    }
    //Read through string to get int
    for ( ; i < line.size() && isdigit(line[i]) ; ++i) {
      ss << line[i];
    } 
  } while ( ss.str() == "" && !eof());
 
  if ( ss.str() != "" )
    return ss.str();
  else
    return "NULL";
}

bool InputReader::eof() {
  return file.eof();
}
