#include "meminstruct.h"

//Initialize RAM with a SIZE
RAM::RAM() {
  arr = new int[SIZE];
}

//Read the addr if its greater than 0
int RAM::read(int addr) {
  return (addr >= 0) ? this->arr[addr] : -1;
}

//Write the address if its greater than 0
void RAM::write(int addr, int val) {
  if (addr >= 0)
    arr[addr] = val;
  //otherwise do nothing
}

#include <iostream>

RAM::~RAM() {
  delete[] arr;
}
