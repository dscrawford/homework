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
Bigram::Bigram(string trainName,bool laplaceSmooth) {
  this->laplaceSmooth = laplaceSmooth;
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
  
  vector<string> L = regexParse(trainText, "\\b.+\\.");
  for (auto l: L) corpuses.push_back(regexParse(l, "\\b[a-zA-Z]+\\b"));
  words = regexParse(trainText, "\\b[a-zA-Z]+\\b");
  sort(words.begin(), words.end());

  cout << "Beginning training..." << endl;
  for (auto word: words) countUnigram[word]++;
  words.erase(unique(words.begin(), words.end()), words.end());
  if (laplaceSmooth) for (auto word: words) countUnigram[word] += words.size();
			     
  for (unsigned long int i = 0; i < corpuses.size(); ++i)
    //Skip first word in corpus, no previous words.
    for (unsigned long int j = 1; j < corpuses[i].size(); ++j) {
      string prevWord = corpuses[i][j-1];
      countBigram[corpuses[i][j]][prevWord]++;
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
  const unsigned long int N = senWords.size();
  unsigned long int countMatrix[N][N];
  float             probMatrix[N][N];
  for (unsigned long int i = 0; i < N; ++i)
    for (unsigned long int j = 0; j < N; ++j) {
      string wordi = senWords[i];
      string wordj = senWords[j];
      countMatrix[i][j] = ((laplaceSmooth) ? countBigram[wordi][wordj] + 1
			   : countBigram[wordi][wordj]);
      probMatrix[i][j]  = (float) countMatrix[i][j] / countUnigram[senWords[i]];
    }
  
  unsigned long int sep = 10;
  for (auto word: senWords)
    sep = (word.length() + 1 > sep) ? word.length() + 1 : sep;

  cout << "Sentence: " << sentence << endl;
  cout << "Counts: " << endl;
  cout << setw(sep) << " ";
  for (auto word: senWords) cout << setw(sep) << word;
  cout << endl;
  for (unsigned long int i = 0; i < N; ++i) {
    cout << endl << setw(sep) << senWords[i];
    for (unsigned long int j = 0; j < N; ++j)
      cout << setw(sep) << countMatrix[i][j];
  }
  cout << endl << endl;
  
  cout << "Probabilities: " << endl;
  cout << setw(sep) << " ";
  for (auto word: senWords) cout << setw(sep) << word;
  cout << endl;
  for (unsigned long int i = 0; i < N; ++i) {
    cout << endl << setw(sep) << senWords[i];
    for (unsigned long int j = 0; j < N; ++j)
      cout << setw(sep) << fixed << (float) probMatrix[i][j];
  }
  cout << endl << endl;

  double senProb = 1;
  for (unsigned long int i = 1; i < N; ++i) senProb *= probMatrix[i][i-1];
  cout << "Probability of sentence: " << senProb << endl;
  return senProb;
}

vector<float> Bigram::predictFile(string testName) {
  vector<float> predictions;
  
  ifstream     testFile(testName);
  stringstream buffer;
  string       testText;
  
  buffer    << testFile.rdbuf();
  testText = buffer.str();

  vector<string> testLines = regexParse(testText, "\\b.+\\.");
  for (auto line: testLines) predictions.push_back(predict(line));
  return predictions;
}


Bigram::~Bigram() {
  words.erase(words.begin(),words.end());
  words.shrink_to_fit();
  corpuses.erase(corpuses.begin(),corpuses.end());
  corpuses.shrink_to_fit();
  countBigram.clear();
}
