// Lab 9b Exercise 1
// Display a row of $ characters based on user input
//
// Program by: Daniel Crawford
#include <iostream>
#include <limits>
using namespace std;
int inputvalidation();
int main() {
	int number;
	number = inputvalidation(); //Gets input from user
	while (number >= 1 && number <= 10) {
		for (int n = number; n >= 1; n--) {
			cout << "$"; // Prints out a dollar sign depending on how large the number is
		}
		cout << "\n";
		number = inputvalidation(); //Gets input from user
	}
	return 0;
}
int inputvalidation() 	//Validates user input
	{
	float number; //Set the number to a float incase the user inputs a decimal
	do{
		cout << "Enter the next number (1-10): ";
		cin >> number;
		number = static_cast<int>(number); //Change the number to an int
		if(cin.fail()) {
			cin.clear();
			cin.ignore(std::numeric_limits<int>::max(),'\n');
			cout << "That is an invalid number.\n";
			number = -1;
			//If invalid input is given then the cin will be cleared and user told to re-enter a number
		}
		else if(number < 0 || number > 10) {
			cout << "That is an invalid number.\n";
			number = -1;
			//If the number is outside the domain then it will ask again
		}
	} while(cin.fail() || number < 0 || number > 10);

	return number;
}
