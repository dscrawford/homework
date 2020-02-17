#ifndef BIGRAM_H
#define BIGRAM_H

#include <map>
#include <vector>
#include <string>

using namespace std;

class Bigram {
 private:
  vector<string>              words;
  vector<vector<string>>      corpuses;
  map<string,int>             countUnigram;
  map<string,map<string,int>> countBigram;
  bool                        laplaceSmooth;

  vector<string>              regexParse(string, string);
 public:
  Bigram(string, bool);
  ~Bigram();
  void          train(string);
  float         predict(string);
  vector<float> predictFile(string);
};

#endif /* BIGRAM_H */
