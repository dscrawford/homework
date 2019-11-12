## Note
   HOMEWORK 3 SOURCE CODE IN LOCATED IN KMeans.java. Not recommended to run other code.
   
   Two files have been included, Homework_3.py and getCompressionRatio.py. Homework_3.py is the
   scikit-learn experiment on the data, expect long running times if executed. getCompressionRatio.py
   will run the java program KMeans for each value of K in 2,5,10,15,20 hundreds of times to get an
   estimate.
## SETUP

Have Koala.jpg and Penguins.jpg in the folder for all the programs to work.

## Run

To run the code, simply just execute with Java installed.

javac KMeans.java
java KMeans <input> <k> <output>

EX:
  java KMeans koala.jpg 3 output.jpg
