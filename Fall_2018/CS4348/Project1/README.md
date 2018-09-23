Made by Daniel Crawford on 09/23/2018
This Program simulates the relationship between a CPU and RAM.
The CPU has a simple 31 instruction long instruction set.

------------------------------------------------------------------------------------------------------------------------
Instructions to run program: 
type "Make"
then "./project1_dsc160130 {inputfile} {timervalue}" to run
-- Make sure you are compiling on a version of g++ that supports std::stoi()(C++11 and up)
-- Make sure to have a IRet instruction at address 1000 in input files if not used for something else.
-- Otherwise, just use a large timer value (1000 or bigger)
------------------------------------------------------------------------------------------------------------------------
Files:
project1_dsc160130(.cc .h): Contains main funciton, and forks the processes. Also creates the pipes here.

cpuinstruct(.cc .h): Contains all the instructions for CPU, runs the programs and the timer.
		     The entire program is really implemented in here.

meminstruct(.cc .h): Contains write and read functions of the RAM. Also holds memory in a 2000 size array.

readinput(.cc .h): Class for reading lines from a file. read() function gets a line that contains an integer

processfunctions.cc: Handles processes for cpu and ram. Initializes RAM and runs CPU, puts RAM in loop

Makefile: Dependencies for compiling the project

sample[1-4].txt: Programs given by class examples

sample5.txt: This program will get a random integer from 1 - 100 and then print a sideways pyramid shape according
	     to that size. The objective of this sample is to test the nested for loops in this simulation.
------------------------------------------------------------------------------------------------------------------------
