#ifndef PROJECT3_H
#define PROJECT3_H

#include <iostream>
#include <fstream>
#include <math.h>

struct pair {
  int beg, end;
};

struct block {
  std::string bytes;
  block() {
    bytes = "";
  }
};

#include "Disk.h"

bool openfstream(std::string, std::fstream&);
void printOptions();
int getInt();
std::string getStr();
void freeContiguous(Disk, int, pair&);
void copyFileFromRealSystemToSimulation(Disk&);
int getFileSize(const char*);
int getBlocks(int);

enum myNums {FILEALLOC, BITMAP};

#define MAXBLOCKSIZE 512
#define MAXBLOCKS 256
#endif //PROJECT3_H
