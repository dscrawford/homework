#include "readinput.h"

InputReader::InputReader(std::string FILENAME) {
  file.open(FILENAME, std::ios::in);
  if (!file) {
    //do something
  }
}

std::string InputReader::read() {
  std::string line;
  std::stringstream  ss;

  do {
    std::getline(file, line);
     
    for (unsigned int i = 0; i < line.size() && isdigit(line[i]) ; ++i) {
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
