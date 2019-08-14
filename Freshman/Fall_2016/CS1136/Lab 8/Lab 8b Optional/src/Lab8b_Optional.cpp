#include <iostream>
#include <cmath>
using namespace std;
void enternumbers(int& number1, int& number2);
int calculatesum(int number1, int number2);

int main() {
	int number1, number2, sum;
	cout << "Enter 0 for both values to end the program\n";
	do {
		enternumbers(number1, number2); //Gets input from user
		sum = calculatesum(number1, number2); //Calculates the sum
		if(sum > 0 || sum < 0) {
			cout << "The sum is " << sum << endl;
			//Displays if the sum is above 0
			}

	} while(number1 != 0 && number2 != 0);
	//Ends when both numbers are equal to 0
	return 0;
	//Repeats the program until the user tells it to end
}
void enternumbers(int& number1, int& number2) {
	cout << "Enter number 1: ";
	cin >> number1;
	cout << "Enter number 2: ";
	cin >> number2;
	//Get input from user

}
int calculatesum(int number1, int number2) {
	int sum;
	if (number1 > number2) {
		for(int i = number2; i <= number1; i++) {
			sum += i;
			}
		return sum;
	}
	else	{
		for(int i = number1; i <= number2; i++) {
					sum += i;
					}
		return sum;
	}
	//Calculates the sum of the values, if statement determines which value is bigger
}
