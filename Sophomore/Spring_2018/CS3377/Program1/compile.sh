#Made by Daniel Crawford(Email: dsc160130@utdallas.edu) on 1/20/2018
#CS 3377.501
echo Setting TEMPDIR environment variable to /scratch
TMPDIR=/scratch
export TMPDIR
echo Compiling file1.cc
g++ -c file1.cc
echo Compiling file2.cc
g++ -c file2.cc
echo Linking files to create executable hw1
g++ file1.o file2.o -o hw1
echo Done
