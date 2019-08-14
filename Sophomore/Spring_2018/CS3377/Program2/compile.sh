#!/bin/bash
#Made by Daniel Crawford(dsc160130@utdallas.edu) on 1/27/2018
#CS3377.501
TMPDIR=/scratch
export TMPDIR
echo Compiling program2.cc into object file program2.o
g++ -I ./include -c program2.cc -o program2.o
echo Compiling fileoperations.cc into object file fileoperations.o
g++ -I ./include -c fileoperations.cc -o fileoperations.o
echo Compiling parsecmd.cc into object file parsecmd.o
g++ -I ./include -c parsecmd.cc -o parsecmd.o
echo Linking files together..
g++ program2.o fileoperations.o parsecmd.o -o program2

