#include "bigram.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <regex>

using namespace std;

Bigram::Bigram(string trainName) {
  ifstream trainFile(trainName);
  stringstream buffer;
  string trainText;
  buffer << trainFile.rdbuf();
  trainText = buffer.str();
  doRegex(trainText, "\\b[a-zA-Z]\\b");
}

void Bigram::doRegex(string trainText, string reg) {
  const regex r(reg);
  smatch sm;

  cout << regex_match(trainText,r) << endl;
  /*
  while (regex_search(trainText, sm, r)) {
    for (auto x:sm) cout << x << " ";
    cout << endl;
  }
  */
  cout << "did i work?" << endl;
}

