#include "project3.h"

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
  if ( str.length() > 9 )
    return NULL;
  return str;
}
