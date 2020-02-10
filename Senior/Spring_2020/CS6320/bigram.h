#ifndef BIGRAM_H
#define BIGRAM_H

#include <map>
#include <vector>
#include <string>

using namespace std;

class Bigram {
 private:
  vector<string>   words;
  map<string, int> wordC;

  void doRegex(string, string);
 public:
  Bigram(string);
};

#endif /* BIGRAM_H */
