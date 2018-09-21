#include "cpuinstruct.h"

CPU::CPU(int* rampipe, int* cpupipe) {
  //Initialize variables and arrays;
  this->SP = USER;
  this->AC = 0;
  this->interruptState = false;
  this->kernelMode = false;
  this->rampipe = rampipe;
  this->cpupipe = cpupipe;
  //Close unused pipes
  close(rampipe[1]);
  close(cpupipe[0]);
}

void CPU::runProgram() {
  //Get all instructions and data from file.
  read(rampipe[0], &(this->timer), sizeof(int));
  curTime = 0;

  //Enter program execution loop
  for (PC = 0; ;) {
    //Send what instruction CPU wants to read
    write(cpupipe[1], &(this->PC), sizeof(int));
    //get Instruction
    read(rampipe[0], &(this->IR), sizeof(int));

    //std::cout << "IR: " << IR << ", PC: " << PC << ", IRSTATE: " << this->interruptState
    //      << ", AC: " << this->AC << std::endl;
    //Run instruction
    usleep(1000);
    PC++;
    runInstruct(IR);
  }
}

//Takes input incase needed for instruction
void CPU::runInstruct(int IR) {
  int isWrite = (IR == 7) ? 1 : 0;
  write(cpupipe[1], &isWrite, sizeof(int));
  //Inform RAM that CPU wants to write this instruction
  switch (IR) {
  case 1:
    loadValue();
    break;
  case 2:
    loadAddr();
    break;
  case 3:
    loadIndaddr();
    break;
  case 4:
    loadIdxXaddr();
    break;
  case 5:
    loadIdxYaddr();
    break;
  case 6:
    loadSpX();
    break;
  case 7:
    storeAddr();
    break;
  case 8:
    get();
    break;
  case 9:
    putPort();
    break;
  case 10:
    addX();
    break;
  case 11:
    addY();
    break;
  case 12:
    subX();
    break;
  case 13:
    subY();
    break;
  case 14:
    copyToX();
    break;
  case 15:
    copyFromX();
    break;
  case 16:
    copyToY();
    break;
  case 17:
    copyFromY();
    break;
  case 18:
    copyToSP();
    break;
  case 19:
    copyFromSP();
    break;
  case 20:
    JumpAddr();
    break;
  case 21:
    JumpIfEqual();
    break;
  case 22:
    JumpIfNotEqual();
    break;
  case 23:
    Call();
    break;
  case 24:
    Ret();
    break;
  case 25:
    IncX();
    break;
  case 26:
    DecX();
    break;
  case 27:
    Push();
    break;
  case 28:
    Pop();
    break;
  case 29:
    Int();
    break;
  case 30:
    IRet();
    break;
  case 50:
    End();
    break;
  }
  curTime++;
  //If IR==0, restore the interrupt state.
  if((curTime == timer && !interruptState) || (this->interruptState && IR == 0)){
    runInterrupt(1000);
  }
}

void CPU::runInterrupt(int PC) {
  //If reverting back from interrupt
  if (this->interruptState) {
    this->PC = popStack();
    this->SP = popStack();
    this->kernelMode = false;
  }
  //If coming from interrupt
  else {
    this->kernelMode = true;
    int tempSP = SP;
    this->SP = this->SYSTM;
    addToStack(tempSP);
    addToStack(this->PC);
    this->PC = PC;
  }
  //Invert the boolean value
  this->interruptState = !(this->interruptState);
  curTime = 0;
}
void CPU::readVals(int& addr, int& val) {
  //Give Ram address
  if (addr > SIZE - 1 || addr < 0) {
    std::cerr << "ERROR: attempting to open address " << addr << " which is out"
      " of range.(IR " << IR << ")" << std::endl;
    _exit(1);
  }
  if (addr > USER && !kernelMode) {
    std::cerr << "ERROR: attempting to open address " << addr << " which is in"
      " system stack.(IR " << IR << ")" << std::endl;
    _exit(1);
  }
  
  write(cpupipe[1], &addr, sizeof(int));
  //Read val
  read(rampipe[0], &val, sizeof(int));

  //Tell RAM that does not request a write
  int isWrite = 0;
  write(cpupipe[1], &isWrite, sizeof(int));
}

//Instruction 1
void CPU::loadValue() {
  readVals(this->PC, this->AC);
  PC++;
}

//Instruction 2
void CPU::loadAddr() {
  int adr;
  //Get address from RAM
  readVals(this->PC, adr);
  //Get new value of AC from RAM
  readVals(adr, this->AC);
  
  PC++;
}

//Instruction 3
void CPU::loadIndaddr() {
  int adr, adr2;
  //Get address from RAM
  readVals(this->PC, adr);
  //Store value at address into AC
  readVals(adr, adr2);
  readVals(adr2, this->AC);

  PC++;
}

//Instruction 4
void CPU::loadIdxXaddr() {
  int adr;
  //Get address from RAM
  readVals(this->PC, adr);
  //Get at this address in RAM
  adr += this->X;
  readVals(adr, this->AC);

  PC++;
}

//Instruction 5
void CPU::loadIdxYaddr() {
  int adr;
  //Get address from RAM
  readVals(this->PC, adr);
  //Get at this address in RAM
  adr += this->Y;
  readVals(adr, this->AC);

  PC++;
}

//Instruction 6
void CPU::loadSpX() {
  int adr = this->SP + this->X;
  readVals(adr, this->AC);

  PC++;
}

//Instruction 7
void CPU::storeAddr() {
  //Request write process from RAM
  int isWrite = 1;
  write(cpupipe[1], &isWrite, sizeof(int));
  int adr;
  //Read the address from the next line
  write(cpupipe[1], &(this->PC), sizeof(int));
  read(rampipe[0], &adr, sizeof(int));
  
  //send address
  write(cpupipe[1], &(adr), sizeof(int));
  //send value in AC now to RAM
  write(cpupipe[1], &(this->AC), sizeof(int));

  PC++;
}

//Instruction 8
void CPU::get() {
  //Generate random number between 1 and 100
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist(1,100);
  this->AC = dist(rng);
}

//Instruction 9
void CPU::putPort() {
  int input;
  readVals(this->PC, input);
  //To pring integer to screen
  if (input == 1) {
    std::cout << this->AC << std::flush;
  }
  //print char to screen
  else if (input == 2) {
    std::cout << (char)this->AC << std::flush;
  }
  else {
    std::cerr << "ERROR: Invalid port" << std::endl;
    _exit(1);
  }

  PC++;
}

//Instruction 10
void CPU::addX() {
  //Subtract X from AC
  this->AC += this->X;
}

//Instruction 11
void CPU::addY() {
  //Subtract Y from AC
  this->AC += this->Y;
}

//Instruction 12
void CPU::subX() {
  //Subtract X from AC
  this->AC -= this->X;
}

//Instruction 13
void CPU::subY() {
  //Subtract Y from AC
  this->AC -= this->Y;
}

//Instruction 14
void CPU::copyToX() {
  //Give X the value in AC
  this->X = this->AC;
}

//Instruction 15
void CPU::copyFromX() {
  //Give AC the value in X
  this->AC = this->X;
}

//Instruction 16
void CPU::copyToY() {
  //Give Y the value in AC
  this->Y = this->AC;
}

//Instruction 17
void CPU::copyFromY() {
  //Give AC the value in Y
  this->AC = this->Y;
}

//Instruction 18
void CPU::copyToSP() {
  //Give SP the value in AC
  this->SP = this->AC;
}

//Instruction 19
void CPU::copyFromSP() {
  //Give AC the value in SP
  this->AC = this->SP;
}

//Instruction 20
void CPU::JumpAddr() {
  int adr;
  //get Address from RAM
  readVals(this->PC, adr);
  this->PC = adr;
  //Update address
}

//Instruction 21
void CPU::JumpIfEqual() {
  if (this->AC == 0)
    JumpAddr();
  else //Otherwise, make sure to skip that next input
    PC++;
}

//Instruction 22
void CPU::JumpIfNotEqual() {
  if (this->AC != 0)
    JumpAddr();
  else //Otherwise, make sure to skip that next input
    PC++;
}

//Instruction 23
void CPU::Call() {
  addToStack(this->PC);     // Current PC + Next Instruction
  JumpAddr();
}

//Instruction 24
void CPU::Ret() {
  //Jump back to instruction after call
  this->PC = popStack();
}

//Instruction 25
void CPU::IncX() {
  this->X++;
}

//Instruction 26
void CPU::DecX() {
  this->X--;
}

//Instruction 27
void CPU::Push() {
  addToStack(this->AC);
}

//Instruction 28
void CPU::Pop() {
  this->AC = popStack();
}

//Instruction 29
void CPU::Int() {
  runInterrupt(1500);
}

//Instruction 30
void CPU::IRet() {
  //Will revert interrupt back to the way it was.
  runInterrupt(0);
}

//Instruction 50
void CPU::End() {
  //end process
  _exit(0);
}

void CPU::addToStack(int input) {
  int temp = -1, isWrite = 1;
  //dummy values
  write(cpupipe[1], &temp, sizeof(int));
  read(rampipe[0], &temp, sizeof(int));

  //Ignore the read, want to write
  write(cpupipe[1], &isWrite, sizeof(int));
  write(cpupipe[1], &temp, sizeof(int));
  if (this->SP >= 0) {
    write(cpupipe[1], &(this->SP), sizeof(int));
    write(cpupipe[1], &input, sizeof(int));
    this->SP--;
  }
  else {
    std::cerr << "ERROR: Stack is full, cannot add(Current instruction:"
	      << IR << ")" << std::endl;
    _exit(1);
  }
}

int CPU::popStack() {
  int val;
  //Simply increment the stack pointer, the data will remain there but
  //will be overwritten on next addToStack
  if (SP < SIZE) {
    this->SP++;
    readVals(this->SP, val);
  }
  else {
    std::cerr << "ERROR: Stack is empty, cannot pop(Current instruction:)"
	      << IR << std::endl;
    _exit(1);
  }
  return val;
}

CPU::~CPU() {}
