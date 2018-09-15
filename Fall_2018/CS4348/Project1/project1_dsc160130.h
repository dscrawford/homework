#ifndef PROJECT1_DSC160130_H
#define PROJECT1_DSC160130_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include "readinput.h"
#include "cpuinstruct.h"
#include "meminstruct.h"

void ramProcess(int, char**, int*, int*);
void cpuProcess(int*, int*);
void cpuInstructs(CPU &, int*, int);
#endif
