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
  map<string,map<string,int>> countM;

  vector<string>              regexParse(string, string);
 public:
  Bigram(string);
  ~Bigram();
  void          train(string);
  float         predict(string);
  vector<float> predictBatch(vector<string>);
};

#endif /* BIGRAM_H */
