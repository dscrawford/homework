#include "meminstruct.h"

RAM::RAM() {
  arr = new int[SIZE];
}

int RAM::read(int addr) {
  return (addr >= 0) ? this->arr[addr] : -1;
}

void RAM::write(int addr, int val) {
  if (addr >= 0)
    arr[addr] = val;
  //otherwise do nothing
}
