#include "bigram.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <regex>
#include <algorithm>

using namespace std;

Bigram::Bigram(string trainName) {
  train(trainName);
}

void Bigram::train(string trainName) {
  ifstream     trainFile(trainName);
  stringstream buffer;
  string       trainText;
  
  buffer << trainFile.rdbuf();
  trainText = buffer.str();
  
  corpuses  = regexParse(trainText, "\\b.+\\n");
  words     = regexParse(trainText, "\\b[a-zA-Z]+\\b");
  sort(words.begin(), words.end());
  words.erase(unique(words.begin(), words.end()), words.end());

  for (unsigned long int i = 0; i < words.size(); ++i) {
    cout << "On word: " << words[i] << endl;
    vector<int> wordCount = findPreviousWordCount(words[i]);
    bigramCounts.push_back(wordCount);

  }
}

vector<string> Bigram::regexParse(string text, string reg) {
  const regex r(reg);
  sregex_token_iterator iter(text.begin(), text.end(), r, 0);
  sregex_token_iterator end;

  vector<string> results;
  for ( ; iter != end; ++iter)
    results.push_back(string(*iter));
  
  return results;
}

vector<int> Bigram::findPreviousWordCount(string word) {
  vector<int> prevWordCount(words.size(), 0);
  for (unsigned long int i = 0; i < corpuses.size(); ++i) {
    vector<string> corpus = regexParse(corpuses[i], "\\b[a-zA-Z]+\\b");
    for (unsigned long int j = 1; j < corpus.size(); ++i) { //Skip first word in corpus, no previous words.
      if (corpus[j] == word) {
	int t = 0;
	string prevWord = corpus[j - 1];
	while (words[t] != prevWord) t++;
	prevWordCount[t]++;
      }
    }
  }
  return prevWordCount;
}
