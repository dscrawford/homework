#include "project3.h"

int getType(std::string input) {
  for (unsigned int i = 0; i < input.length(); ++i) {
    input[i] = std::tolower(input[i]);
  }

  if (input == "contiguous")
    return CONTIGUOUS;
  else if (input == "indexed")
    return INDEXED;
  else if (input == "chained")
    return CHAINED;

  return -1;
}

int getInt() {
  std::string str;
  std::getline(std::cin, str);
  for (unsigned int i = 0; i < str.length(); ++i) {
    if (!std::isdigit(str[i]))
      return -1;
  }

  return std::stoi(str);
}

std::string getStr() {
  std::string str;
  std::getline(std::cin, str);
  return str;
}
