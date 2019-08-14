// Lab 8b Exercise 2
// Calculate investment amount
//
// Program by: Daniel Crawford
#include <iostream>
#include <limits>
#include <iomanip>
using namespace std;

//Prototype functions
double initialinvestment();
double annualinterestrate();
int numberofmonths();
double investmentreturn(double ratebymonth, double investment, int months);
void displaytotals(double investmentbalance, int months, int investment);

int main() {
	int months;
	double  ratebymonth, investment, interest, investmentbalance;
	//Defines the variables, months in the int, all else are doubles
	investment = initialinvestment();
	interest = annualinterestrate();
	months = numberofmonths();
	//Three assignment functions are input from the user
	ratebymonth = ( (interest / 12) / 100);
	//Assigns ratebymonth to the result
	investmentbalance = investmentreturn(ratebymonth, investment, months);
	//Calculates the final value of the investment
	displaytotals(investmentbalance, months, investment);
	//Displays the results
	return 0;
}

double initialinvestment() {
	double investment;
	cout << "The minimum investment is 10000.00" << endl;
	do
	{
		cout << "Enter the amount of the investment: ";
		cin >> investment;
		if (investment < 10000) {
			cout << "Error, the minimum investment is 10000.00" << endl;
		}
		if (cin.fail()) {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(),'\n');
			cout << "Invalid input.";
		}
	} while (investment < 10000);
	return investment;
	//Validates the user input, then returns a value above 10000
}
double annualinterestrate() {
	double interest;
	do {
		cout << "Enter the annual interest rate: ";
		cin >> interest;
		if (cin.fail()) {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(),'\n');
			cout << "Invalid Input." << endl;
		}
		else if (interest < 0) {
			cout << "Error, the interest rate should not be negative" << endl;
		}

	} while (interest < 0);
	return interest;
	//Validates the user input, then returns a value equal to or above 0
}
int numberofmonths() {
	int months;
	do {
		cout << "Enter the number of months of the investment: ";
		cin >> months;
		if (cin.fail()) {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(),'\n');
			cout << "Invalid Input." << endl;
		}
		else if (months < 0) {
			cout << "Error, the number of months should not be negative" << endl;
		}

	} while (months < 0);
	return months;
	//Validates the user input, then returns a value equal to or above 0
}
double investmentreturn(double ratebymonth, double investment, int months) {
	double interestearned;
	do {
		interestearned = (investment * ratebymonth);
		investment += interestearned;
		months -= 1;
	} while (months > 0);
	return investment;
	//Calculates the investment return after it adds up each month
}

void displaytotals(double investmentbalance, int months, int investment) {
	cout << "After " << months << " months your investment of $" << investment << " will be worth $" << fixed << setprecision(2) << investmentbalance;
	//displays the results.
}
