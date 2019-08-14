// Assignment 1 Part 1 for CS 1336.001
// Programmer: Daniel Crawford
// This program will tell you how the best in cost between each phone plan.

#include <iostream> // std::cout, std::cin...
#include <iomanip> // std::setw, std::setprecision, std::fixed
using namespace std; // dont have to write std:: before every statement

int main() {
	double numberofGBofdata, numberofphonelines;
	double costperlineplan1, costperGBplan1, plan1linecost, plan1GBcost, plan1cost;
	double costperlineplan2, costperGBplan2, plan2linecost, plan2GBcost, plan2cost;
	double costperlineplan3, costperGBplan3, plan3linecost, plan3GBcost, plan3cost;
	// defining all variables as a double
	costperlineplan1 = 23;
	costperlineplan2 = 15;
	costperlineplan3 = 12;
	costperGBplan1 = 10.00;
	costperGBplan2 = 11.00;
	costperGBplan3 = 12.00;
	// setting literals which will define the difference between plans 1, 2 and 3
	cout << "Enter the total needed GB of data: ";
	cin >> numberofGBofdata;
	cout << "Enter the number of phone lines: ";
	cin >> numberofphonelines;
	// asks for input of variables numberofGBofdata and numberofphonelines

	plan1linecost = costperlineplan1 * numberofphonelines;
	plan2linecost = costperlineplan2 * numberofphonelines;
	plan3linecost = costperlineplan3 * numberofphonelines;
	// multiplies costperline and numberofphonelines to create planxlinecost
	plan1GBcost = costperGBplan1 * numberofGBofdata;
	plan2GBcost = costperGBplan2 * numberofGBofdata;
	plan3GBcost = costperGBplan3 * numberofGBofdata;
	// multiplies costperGBplanx and numberofGBofdata to create planxGBcost
	plan1cost = plan1linecost + plan1GBcost;
	plan2cost = plan2linecost + plan2GBcost;
	plan3cost = plan3linecost + plan3GBcost;
	// adds planxlinecost and planxGBcost to create the final cost of the plan "planxcost"
	cout << "\n\n\n";
	// creates some space
	cout << fixed << setprecision(2);
	// sets the decimal precision to two decimal points
	cout << setw(20) << left << "Plan 1 cost is:" << setw(5) << right << plan1cost << endl;
	cout << setw(20) << left << "Plan 2 cost is:" << setw(5) << right << plan2cost << endl;
	cout << setw(20) << left << "Plan 3 cost is:" << setw(5) << right << plan3cost << endl;
	// prints out the cost for each plan
	return 0;
	// exit value of zero
}
