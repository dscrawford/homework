//Calculator program
//
//
//Program by: Daniel Crawford
#include <iostream>
#include <limits>
#include <fstream>
#include <math.h>
#include <stdio.h>

using namespace std;

int getCommand();
double processCommand(int command, double oldValue, ostream &output);
void displaymenu();
double inputvalidation();
double inputvalidationdivide(ostream &output);

int main() {
	int command;
	double oldvalue = 0;
	double total = 0;

	ofstream outputFile;
	const string OUTPUT_FILE_NAME = "calculator.txt";
	outputFile.open(OUTPUT_FILE_NAME);
	if(!outputFile.fail()) {
		cout << "The output file " << OUTPUT_FILE_NAME << " was opened." << endl;
	}
	else {
		cout << "The output file " << OUTPUT_FILE_NAME << " was not opened." << endl;
	}

	do {

		outputFile << "total " << total << endl;  //Saves to file the total amount to the file
		cout << endl << "Total: " << total << endl << endl;//Prints out the total

		command = getCommand();//Asks for and receives users input
		oldvalue = processCommand(command, oldvalue, outputFile);//Assigns oldvalue to the result of the arithmetic
		total = oldvalue;//New value becomes old

	} while (command != 0); //Continously loops until user wants to stop
	outputFile.close();
	return 0;
}

void displaymenu() { //Displays the menu
	cout << "0. Quit the application" << endl;
	cout << "1. Add" << endl;
	cout << "2. Subtract" << endl;
	cout << "3. Multiply" << endl;
	cout << "4. Divide" << endl;
	cout << "5. Clear total" << endl;
	cout << "6. Square root of the current total" << endl;
	cout << "7. Current total to the power of x" << endl;
}

double processCommand(int command, double oldValue, ostream &output) {
	double newValue, value; //The value that will be returned and a filler value
	switch (command) {
		case 0:
			output << "exit" << endl;
			break;
		case 1:
			cout << "Enter the value to be added: ";
			value = inputvalidation();
			newValue = oldValue + value; //Asks for input then does addition
			output << "add " << value << endl;  //Saves to file the arithmetic type and result
			break;
		case 2:
			cout << "Enter the value to be subtracted: ";
			value = inputvalidation(); //Asks for input then does subtraction
			newValue = oldValue - value;
			output << "subtract " << value << endl;  //Saves to file the arithmetic type and result
			break;
		case 3:
			cout << "Enter the value to be multiplied: ";
			value = inputvalidation(); //Asks for input then does multiplication
			newValue = oldValue * value;
			output << "multiply " << value << endl;  //Saves to file the arithmetic type and result
			break;
		case 4:
			value = inputvalidationdivide(output); //Asks for input then does division
			newValue = oldValue / value;

			break;
		case 5:
			newValue = 0;
			output << "clear" << endl; //Clears the total then saves it to the file
			break;
		case 6:
			newValue = pow(oldValue, .5); //Finds the square root
			output << "sqrt " << endl; //Saves to file the arithmetic type and result
			break;
		case 7:
			cout << "Enter the value for x: "; //Asks for input then finds the power
			value = inputvalidation();
			newValue = pow(oldValue, value);
			output << "pow " << value << endl; //Saves to file the arithmetic type and result
			break;
		default:
			cout << "Bad input.\n"; //If the user somehow gets here
			break;
		}
	return newValue;
}

int getCommand() {
	int number;
	do {
		cout << "Enter one of the following options: \n";
		displaymenu();
		cin >> number;
		if(number < 0 || number > 7) {
			cout << "The menu item you entered is invalid.\n";
		}
		else if (cin.fail()){
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(),'\n');
			cout << "The menu item you entered is invalid.\n";
			number = -1;
		}
	} while(number < 0 || number > 7);
	//Validates the user input then returns the value
	return number;
}
double inputvalidation() {
	double number;
	bool validation;
	do {
		cin >> number;
		if (cin.fail()){
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(),'\n');
			cout << "Bad input\n";
			validation = false;
		}
		else {
			validation = true;
		}
	} while(validation == false);
	//Validates the user input then returns the value
	return number;
}
double inputvalidationdivide(ostream &output) {
	double number;
	bool validation;
	do {
		cout << "Enter the value of the divisor: ";
		cin >> number;
		if (cin.fail()){
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(),'\n');
			cout << "Bad input\n";
			validation = false;
		}
		else if(number == 0) {
			validation = true;
			cout << "Division by 0 is not allowed\n";
			output << "Divide 0 not allowed " << endl; //Saves to file that the division was not allowed
			number = 1;
		}
		else {
			validation = true;
			output << "divide " << number << endl; //Saves to file the arithmetic type and result
		}
	} while(validation == false);
	//Validates the user input then returns the value, returns 1 if they tried to divide by zero
	return number;
}
