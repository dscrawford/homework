#ifndef PROJECT3_H
#define PROJECT3_H

#include <iostream>
#include <fstream>
#include <math.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <string.h>
#include <cstdlib>
#include <iomanip>

struct pair {
  int beg, end;
  pair() {
    beg = -1;
    end = -1;
  }
};

struct block {
  std::string bytes;
  block() {
    bytes = "";
  }
};

#include "Disk.h"
#include "Logic.h"

bool openfstream(std::string, std::fstream&);
void printOptions();
int getInt();
std::string getStr();
int getType(std::string);

enum myNums {FILEALLOC, BITMAP, CONTIGUOUS, CHAINED, INDEXED};

#define MAXBLOCKSIZE 512
#define MAXBLOCKS 256
#endif //PROJECT3_H
