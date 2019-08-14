// Lab 8b Exercise 1
// Calculate the product
//
// Program by: Daniel Crawford
#include <iostream>
using namespace std;
// Function prototypes
int getNumber(); // Read in a number
void displayProduct(int product); // display the product
int main()
{
	int number, product = 1, count = 0;
	// Get the first number
	do
	{
		// While there are still more numbers to process
		// Calculate the product (so far)
		number = getNumber(); // get the next number (or 0 to stop)
		//moved the getNumber() to inside the do while loop only
		if (number == 0) {
			break;
		}
		//If the user enters 0, end the statement
		product = product * number;
		count++; // keep track of how many we've processed

	} while (number != 0); //Tests
// Display the product (if at least one input number was not 0)
	if (count > 0)
		{
		displayProduct(product);
		}
}
// Prompt the user to enter in a number
int getNumber()
{
	int number;
	cout << "Enter an integer number to be included in the product"	<< endl << "or enter 0 to end the input: ";
	cin >> number;
	return number;
}
// Display the product we've computed.
void displayProduct(int product)
{
	cout << endl << "The product is " << product << "." << endl;
}
