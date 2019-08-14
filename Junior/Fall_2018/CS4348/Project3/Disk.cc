#include "project3.h"

Disk::Disk() {
  data = new block[MAXBLOCKS];
}

block Disk::read(int i) {
  return data[i];
}

void Disk::write(block b, int i) {
  if (b.bytes.length() > MAXBLOCKSIZE) {
    std::cerr << "ERROR: Block too large" << std::endl;
  }
  data[i] = b;
}
