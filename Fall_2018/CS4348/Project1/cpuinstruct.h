#ifndef CPUINSTRUCT_H
#define CPUINSTRUCT_H

#include <unistd.h>
#include <memory>
#include <string>
#include <iostream>
#include <cstdlib>
#include <random>

class CPU {
private:
  //Registers
  int PC;
  int SP;
  int IR;
  int AC;
  int X;
  int Y;

  //Checks whether interrupt is currently happening
  bool interruptState;
  //How many instructions to run before interrupt
  int timer;
  int curTime;

  bool kernelMode;
  
  //Pipes for communicating with RAM
  int* rampipe;
  int* cpupipe;

  //Size of stacks
  const static int SIZE  = 2000;
  const static int SYSTM = 1999;
  const static int USER  = 999;

  void runInstruct(int);//Run an instruction

  void runInterrupt(int);//Run interrupt state

  //Add value to stack
  void addToStack(int);

  //Pop a value from stack
  int popStack();

  //Function for loading a value from RAM
  void readVals(int&, int&);

  //
  //CPU Instructions
  //

  //Load the value into the AC
  void loadValue();

  //Load the value at the address into the AC
  void loadAddr();

  //Load the value from the address found in the given address into the AC
  void loadIndaddr();
  
  //Load the value at(address+X) into the AC
  void loadIdxXaddr();       

  //Load the value at(address+Y) into the AC
  void loadIdxYaddr();

  //Load from (Sp+X) into the AC
  void loadSpX();

  //Store the value in the AC into the address
  void storeAddr();          

  //Gets a random int from 1 to 100 into the AC
  void get();

  //If port=1, writes AC as an int to the screen
  //If port=2, writes AC as a char to the screen
  void putPort();

  //Add the value in X to the AC
  void addX();

  //Add the value in Y to the AC
  void addY();              

  //Subtract the value in X from the AC
  void subX();

  //Subtract the value in Y from the AC
  void subY();               

  //Copy the value in the AC to X
  void copyToX();            

  //Copy the value in X to the AC
  void copyFromX();

  //Copy the value in the AC to Y
  void copyToY();

  //Copy the value in Y to the AC
  void copyFromY();          

  //Copy the value in the AC to the SP
  void copyToSP();           

  //Copy the value in SP to the AC
  void copyFromSP();         

  //Jump to the addr
  void JumpAddr();           

  //Jump to the address only if the value in the AC=0
  void JumpIfEqual();

  //Jump to the address only if the value in the AC!=0
  void JumpIfNotEqual();
  
  //Push return address onto stack, jump to the address
  void Call();               

  //Pop return address from the stack, jump to the address
  void Ret();

  //Increment the value in X
  void IncX();               

  //Decrement the value in X
  void DecX();              

  //Push AC onto stack
  void Push();              

  //Pop from stack into AC
  void Pop();

  //Perform system call
  void Int();             

  //Return from system call
  void IRet();         

  //End execution
  void End();

  //
  //End CPU Instruction
  //

public:
  CPU(int* rampipe, int* cpupipe);

  void runProgram();//Run entire program

  ~CPU();
};

#endif // CPUINSTRUCT_H
