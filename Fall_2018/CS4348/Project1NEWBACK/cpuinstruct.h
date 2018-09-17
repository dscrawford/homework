#ifndef CPUINSTRUCT_H
#define CPUINSTRUCT_H

#include "meminstruct.h"
#include <unistd.h>
#include <memory>

class CPU {
private:
  //Registers
  int PC;
  int SP;
  int IR;
  int AC;
  int X;
  int Y;
  //Stack
  int* userstck;
  int* systemstck;
  //Checks whether interrupt is currently happening
  bool interruptState;
  const static int SIZE = 2000;

  void loadValue(int);    //Load the value into the AC
  
  void loadAddr(int*, int);   //Load the value at the address into the AC
  
  void loadIndaddr();        //Load the value from the address found in the given
                             //address into the AC
  
  void loadIdxXaddr();       //Load the value at(address+X) into the AC
  
  void loadIdxYaddr();       //Load the value at(address+Y) into the AC
  
  void loadSpX();            //Load from (Sp+X) into the AC
  
  void storeaddr();          //Store the value in the AC into the address
  
  void get();                //Gets a random int from 1 to 100 into the AC
  
  void putport();            //If port=1, writes AC as an int to the screen
                             //If port=2, writes AC as a char to the screen
  
  void addX();               //Add the value in X to the AC
  
  void addY();               //Add the value in Y to the AC
  
  void subX();               //Subtract the value in X from the AC
  
  void subY();           //Subtract the value in Y from the AC
  
  void copyToX();        //Copy the value in the AC to X
  
  void copyFromX();      //Copy the value in X to the AC
  
  void copyToY();        //Copy the value in the AC to Y
  
  void copyFromY();      //Copy the value in Y to the AC
  
  void copyToSp();       //Copy the value in the AC to the SP
  
  void copyFromSp();     //Copy the value in SP to the AC
  
  void JumpAddr();       //Jump to the addr
  
  void JumpIfEqual();    //Jump to the address only if the value in the AC=0
  
  void JumpIfNotEqual(); //Jump to the address only if the value in the AC!=0
  
  void Call();           //Push return address onto stack, jump to the address
  
  void Ret();            //Pop return address from the stack, jump to the address
  
  void IncX();           //Increment the value in X
  
  void DecX();           //Decrement the value in X
  
  void Push();           //Push AC onto stack
  
  void Pop();            //Pop from stack into AC
  
  void Int();            //Perform system call
  
  void IRet();           //Return from system call
  
  void End();            //End execution
public:
  CPU();

  void runInstruct(int*, int);

  void getInstructs(int *, int);

  void addToStack(int);
  ~CPU();
};

#endif // CPUINSTRUCT_H
