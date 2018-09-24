//Made by Daniel Crawford on 2/19/2018(dsc160130@utdallas.edu)
//Section 3377.501
#include "program3.h"

std::string PATH = "/home/daniel/Documents/cpp_programs/Program3/bin/gawk";

int main() {
  //Display path to gawk
  std::cout << "gawk at: " << PATH << std::endl;
  //Get the version information
  std::cout << "Shellcmd1: " << PATH << " --version" << std::endl;
  std::string gawkVersion = getgawkcommand("--version");
  //Get the results from the gawk program on "numbers.txt"
  std::cout << "Shellcmd2: " << PATH << " -f gawk.code numbers.txt" << std::endl;
  std::string results = getgawkcommand("-f gawk.code numbers.txt");

  int result1, result2;
  //Parse the results from the string received from gawk
  getResults(result1, result2, results);
  //Display all data gathered from gawk program
  std::cout << "The first call to gawk returned: \n\n" << gawkVersion << std::endl
            << "The second call to gawk returned: " << results << std::endl
            << "The sum of Column 1 is: " << result1 << std::endl
            << "The sum of Column 4 is: " << result2 << std::endl
            << "The Sum of the two numbers is: " << result1 + result2 << std::endl;
  return 0;
}
