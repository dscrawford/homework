#include "meminstruct.h"

RAM::RAM() {
  arr = new int[SIZE];
}

int RAM::read(int addr) {
  return this->arr[addr];
}

void RAM::write(int addr, int val) {
  arr[addr] = val;
}
