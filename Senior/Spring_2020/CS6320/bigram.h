#ifndef BIGRAM_H
#define BIGRAM_H

#include <map>
#include <vector>
#include <string>

using namespace std;

class Bigram {
 private:
  vector<string>      corpuses;
  vector<string>      words;
  vector<vector<int>> bigramCounts;

  vector<string> regexParse(string, string);
  vector<int> findPreviousWordCount(string);
 public:
  Bigram(string);
  void train(string);
};

#endif /* BIGRAM_H */
