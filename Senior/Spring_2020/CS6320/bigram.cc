#include "bigram.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <regex>
#include <algorithm>
#include <iomanip>

using namespace std;

/* Bigram
 * Constructor acts as a shortcut to the train function.
 *
 */
Bigram::Bigram(string trainName) {
  train(trainName);
}

/* train
 * Given a filename 'trainName' this will read the file and then extract its
 * contents. Then, it will construct the bigram(count table).
 */
void Bigram::train(string trainName) {
  ifstream     trainFile(trainName);
  stringstream buffer;
  string       trainText;
  
  buffer    << trainFile.rdbuf();
  trainText = buffer.str();
  
  vector<string> lines = regexParse(trainText, "\\b.+\\n");
  for (unsigned long int i = 0; i < lines.size(); ++i)
    corpuses.push_back(regexParse(lines[i], "\\b[a-zA-Z]+\\b"));
  
  words = regexParse(trainText, "\\b[a-zA-Z]+\\b");
  sort(words.begin(), words.end());
  words.erase(unique(words.begin(), words.end()), words.end());

  cout << "Beginning training..." << endl;
  for (unsigned long int i = 0; i < corpuses.size(); ++i) {
    //Skip first word in corpus, no previous words.
    for (unsigned long int j = 1; j < corpuses[i].size(); ++j) {
      string prevWord = corpuses[i][j-1];
      countM[corpuses[i][j]][prevWord]++;
    }
  }

  cout << "Done!" << endl;
}

/* regexParse
 * Given a string and a regular expression, this will split/parse
 * the string into a vector of strings based on the regular expression.
 */
vector<string> Bigram::regexParse(string text, string reg) {
  const regex r(reg);
  sregex_token_iterator iter(text.begin(), text.end(), r, 0);
  sregex_token_iterator end;

  vector<string> results;
  for ( ; iter != end; ++iter)
    results.push_back(string(*iter));
  
  return results;
}

float Bigram::predict(string sentence) {
  vector<string> senWords = regexParse(sentence, "\\b[a-zA-Z]+\\b");
  unsigned long int n = senWords.size();
  unsigned long int sep = 5;
  for (unsigned long int i = 0; i < n; ++i)
    sep = (senWords[i].length() + 1 > sep) ? senWords[i].length() + 1 : sep;

  cout << setw(sep) << " ";
  for (unsigned long int i = 0; i < n; ++i) {
    cout << setw(sep) << senWords[i];
  }
  cout << endl;
  for (unsigned long int i = 0; i < n; ++i) {
    cout << setw(sep) << senWords[i];
    for (unsigned long int j = 0; j < n; ++j) {
      cout << setw(sep) << countM[senWords[i]][senWords[j]];
    }
    cout << endl;
  }
  cout << "I'm done!" << endl;
  return 0.0f;
}

Bigram::~Bigram() {
  words.erase(words.begin(),words.end());
  words.shrink_to_fit();
  corpuses.erase(corpuses.begin(),corpuses.end());
  corpuses.shrink_to_fit();
  countM.clear();
}
