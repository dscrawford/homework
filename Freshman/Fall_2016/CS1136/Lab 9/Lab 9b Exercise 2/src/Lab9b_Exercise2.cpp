// Lab 9b Exercise 2
// Percentages of numbers input from a file.
//
// Program by: Daniel Crawford
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
using namespace std;


void calculatepercentages(ifstream& inputFile);

int main() {
	const string INPUT_FILE_NAME = "numbers.txt";
	ifstream inputFile;
	inputFile.open(INPUT_FILE_NAME);//Re-opens input file to reuse the value
	calculatepercentages(inputFile);

	return 0;
}

void calculatepercentages(ifstream& inputFile) {
	double negativeCount = 0.0, AboveCount = 0.0, EqualCount = 0.0;
	int value, count;
	double percentageN, percentageA, percentageE;
	while (inputFile >> value) {
		count++;
		if(value < 0) {
			negativeCount++;
		}
		else if(value > 0) {
			AboveCount++;
		}
		else {
			EqualCount++;
		}
		//Counts and sends each number to a category
	}
	percentageN = (negativeCount / count) * 100;
	//Calculates the percentage of negative numbers
	cout << "There were " <<  fixed << setprecision(1) << percentageN << "% negative numbers.\n";
	percentageA = (AboveCount / count) * 100;
	//Calculates the percentage of positive numbers
	cout << "There were " <<  fixed << setprecision(1) << percentageA << "% numbers greater than zero.\n";
	percentageE = (EqualCount / count) * 100;
	//Calculates the percentage of numbers equal to 0
	cout << "There were " << fixed << setprecision(1) << percentageE << "% numbers equal to zero.\n";
	//Displays the numbers equal to 0
	inputFile.close();

}
