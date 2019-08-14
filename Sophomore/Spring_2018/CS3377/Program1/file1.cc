//Made by Daniel Crawford(Email: dsc160130@utdallas.edu) on 1/20/2018
//cs 3377.501
#include <iostream>
void func1();

int main(int argc, char* argv[]) {
  //Display the amount of arguments sent into the program
  std::cout << "argc was: " << argc << std::endl;
  //DIsplay all the arguments sent.
  for (int i = 0; i < argc; ++i) {
    std::cout << argv[i] << std::endl;
  }
  //Call func1() from file2.cc
  func1();
}
