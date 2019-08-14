//Made by Daniel Crawford on 2/19/2018(dsc160130)
//Section 3377.501
#include "program3.h"
#include <sstream>
int parseInt(std::string);

void getResults(int& x1, int& x2, std::string results) {
  //Stream will obtain results from gawk and parse them into ints.
  std::stringstream ss(results);
  std::string result;

  //Read first columns results
  std::getline(ss, result, ' ');
  x1 = parseInt(result);
  //Read second coulmns results
  std::getline(ss, result, ' ');
  x2 = parseInt(result);
}
int parseInt(std::string str) {
  //Parse the result to an integer with atoi, if 0 is returned, inform the user and exit program
  int result = atoi(str.c_str());
  if (result == 0) {
    std::cerr << "error: either entire sum was 0 or string was not convertable to an int"
              << std::endl;
  }
  return result; 
}
