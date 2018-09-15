This example shows how to create a small matrix using CDK.

It provides this info:
   1) CDK Setup
   2) Creating a CDK Matrix
   3) Placing txt info into a matrix box

Please note these items to students:
   1) You must have preivously compiled/installed CDK
   2) -I and -L flags to compiler have locations of CDK headers and libraries
   3) Linking with the CDK and curses libraries via these commpiler flags: -lcdk -lcurses
   4) The arrays of characters need to be defined as (const char **) However, the newCDKMatrix wants (char **).  Note manual cast.
   5) A matrix starts at location 1,1.  Arrays start at location 0.  Therefore arrays are one longer than they should be and the [0] elements are not used.


