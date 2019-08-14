//Calculator program
//
//
//Program by: Daniel Crawford

#include <iostream>
#include <limits>
#include <fstream>
#include <math.h>

using namespace std;

int getCommand();
double processCommand(int command, double oldValue);
void displaymenu();
double inputvalidation();
double inputvalidationdivide();

int main() {
	int command;
	double oldvalue = 0;
	double total = 0;
	do {
		cout << endl << "Total: " << total << endl << endl; //Prints out the total
		command = getCommand(); //Asks for and receives users input
		oldvalue = processCommand(command, oldvalue); //Assigns oldvalue to the result of the arithmetic
		total = oldvalue;

	} while (command != 0); //Continously loops until user wants to stop
	return 0;
}

void displaymenu() {
	cout << "0. Quit the application" << endl;
	cout << "1. Add" << endl;
	cout << "2. Subtract" << endl;
	cout << "3. Multiply" << endl;
	cout << "4. Divide" << endl;
	cout << "5. Clear total" << endl;
	cout << "6. Square root of the current total" << endl;
	cout << "7. Current total to the power of x" << endl;
}

double processCommand(int command, double oldValue) {
	double newValue;
	switch (command) {
		case 0:
			break;
		case 1:
			cout << "Enter the value to be added: ";
			newValue = oldValue + inputvalidation();
			break;
		case 2:
			cout << "Enter the value to be subtracted: ";
			newValue = oldValue - inputvalidation();
			break;
		case 3:
			cout << "Enter the value to be multiplied: ";
			newValue = oldValue * inputvalidation();
			break;
		case 4:
			newValue = oldValue / inputvalidationdivide();
			break;
		case 5:
			newValue = 0;
			break;
		case 6:
			newValue = pow(oldValue, .5);
			break;
		case 7:
			cout << "Enter the value for x: ";
			newValue = pow(oldValue, inputvalidation());
			break;
		default:
			cout << "Bad input.\n";
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
	return number;
}
double inputvalidationdivide() {
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
			number = 1;
		}
		else {
			validation = true;
		}
	} while(validation == false);
	return number;
}
