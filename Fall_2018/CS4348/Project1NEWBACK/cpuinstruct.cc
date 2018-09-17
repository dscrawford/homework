#include "cpuinstruct.h"
#include <iostream>

CPU::CPU() {
  userstck = new int[SIZE];
  systemstck = new int[SIZE];
  this->PC = this->SIZE - 1;
  this->SP = this->SIZE - 1;
  this->interruptState=false;
}

void CPU::addToStack(int value) {
}

void CPU::getInstructs(int* mypipe, int instrc) {
  //Instruction
  int instr;
  //Get all instructions and data from file.
  for (int i = 0; i < instrc; ++i) {
    read(mypipe[0], &instr, sizeof(int));
    std::cout << "cpu(instr): " << instr << std::endl;
  } 
}
  
void CPU::runInstruct(int* mypipe, int instr) {
  switch (instr) {
  case 1: //Load value from next instruction, then into AC
    int x;
    read (mypipe[0], &x, sizeof(int));
    loadValue(x);
    break;
  case 2:
    loadAddr(mypipe,instr);
    break;
  case 3:
    break;
  case 4:
    break;
  case 5:
    break;
  case 6:
    break;
  case 7:
    break;
  case 8:
    break;
  case 9:
    break;
  case 10:
    break;
  case 11:
    break;
  case 12:
    break;
  case 13:
    break;
  case 14:
    break;
  case 15:
    break;
  case 16:
    break;
  case 17:
    break;
  case 18:
    break;
  case 19:
    break;
  case 20:
    break;
  case 21:
    break;
  case 22:
    break;
  case 23:
    break;
  case 24:
    break;
  case 25:
    break;
  case 26:
    break;
  case 27:
    break;
  case 28:
    break;
  case 29:
    break;
  case 30:
    break;
  case 50:
    break;
  }
}

void CPU::loadValue(int value) {
  this->AC = value;
}

void CPU::loadAddr(int mypipe[], int address) {
  int value;
  //indicate to memory that cpu wants something
  write(mypipe[1], &address, sizeof(int));
  //read value found from memory
  read(mypipe[0], &value, sizeof(int));
  loadValue(value);
}

CPU::~CPU() {}
