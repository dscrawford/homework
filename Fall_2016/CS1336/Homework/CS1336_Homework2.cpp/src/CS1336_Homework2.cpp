// Factorial Program
//
//
// Program by: Daniel Crawford


// Calculate factorials from 1! to 20!
//
// n! is 1 * 2 * 3 * ... * n-2 * n-1 * n
//
// Calculate and display to cout and write out to a file all of the values from 1! to 20!
//
// You will need to calculate your factorials in a long long int or an unsigned long long int.
// 21! will not fit into either a long long int or an unsigned long long int.
//
// Your program must be using C++11 to get the unsigned long long int or long long int to work.
//
// If you are using Visual Studio 2013 or later you will be OK. If you are using Eclipse,
// xCode, Code::Blocks or some other G++ based IDE you will need to add the -std=c++11 flag to
// the compiler options.
//
#include <iostream>
#include <fstream>

using namespace std;

// function prototypes
// calculate the factorial (up to 20)
unsigned long long int calculateFactorial(int value);
// display the results
void display(ostream &output, int value, unsigned long long int valueFactorial);
// The main drives the application
//
// First your program needs to open an output file. You must check to make sure the open worked. If the file
// did not open you need to output an error message and exit the program.
//
// Your main funciton will have a processing loop that will drive the caluclation of all of the factorials for the
// numbers 1 through 20.
//
// For each new number you will call calculateFactorial to calculate the actual factorial for the number.
//
// You will call the display function and have it output the number and the number's factorial. You will also
// pass the display function a ostream object (either cout or your outputFile).
//
// You need to call the display twice each time through your processing loop. Once you will pass it cout and the
// second time you will pass it the output file.
//
// After the loop has completed you need to make sure you close your output file.
int main()
{

	const string OUTPUT_FILE_NAME = "factorial.txt";
	ofstream outputFile;
	unsigned long long int valueFactorial;
	outputFile.open(OUTPUT_FILE_NAME); //Creates a file factorial.txt then opens it
	if(!outputFile.fail()) {
		cout << "The file " << OUTPUT_FILE_NAME << " was opened." << endl; //Informs the user the file was opened.
		for(int number = 1; number <= 20; number++) {
			valueFactorial = calculateFactorial(number); //Creates 20 numbers and finds their factorials
			display(outputFile, number, valueFactorial); //Puts the data into the .txt file
			display(cout, number, valueFactorial); //Displays it to the user
		}

		}
	else {
		cout << "The file " << OUTPUT_FILE_NAME << " could not be opened." << endl; //Informs the user the file could not be opened.
	}
	// PUT YOUR MAIN PROCESSING LOOP HERE
	// here is some pseudo-code:
	// open the output file. If it does not open output a message and end the program
	//
	// for number from 1 to 20
	//    numberFactorial = calculateFactorial(number)
	//    display(cout, number, numberFactorial)
	//    display(outputFile, number, numberFactorial)
	// end for
	//
	// close the output file
}

// calculate n! for the case where n is value.
 unsigned long long int calculateFactorial(int value)
{
	unsigned long long int factorial = 1; //Sets factorial equal to 1
	for(int number = 1; number <= value; number++) {
		factorial *= number; //Multiplies the number until number is equal to value
	}
	return factorial; //Returns the factorial
}

// display the results
void display(ostream &outputFile, int value, unsigned long long int valueFactorial)
{
	outputFile << "Number: " << value << endl
	<< "Factorial: " << valueFactorial << endl;
	outputFile << endl;
	//Displays or writes the results

}

