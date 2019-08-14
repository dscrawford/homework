/*
 * dsc160130_CS1336A1P2.cpp
 *
 *  Created on: Sep 19, 2016
 *      Author: dsc160130
 */
// Assignment 1 Part 2 for CS 1336.001
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
	double costperlineplan4, costperGBplan4, plan4linecost, plan4GBcost, plan4cost;
	// defining all variables as a double
	costperlineplan1 = 15.00;
	costperlineplan2 = 11.95;
	costperlineplan3 = 23.50;
	costperlineplan4 = 189.75;
	costperGBplan1 = 11.50;
	costperGBplan2 = 12.95;
	costperGBplan3 = 9.95;
	costperGBplan4 = 0;
	// setting literals which will define the difference between plans 1, 2, 3 and 4
	cout << "Enter the total needed GB of data: ";
	cin >> numberofGBofdata;
	cout << "Enter the number of phone lines: ";
	cin >> numberofphonelines;
	// asks for input of variables numberofGBofdata and numberofphonelines

	plan1linecost = costperlineplan1 * numberofphonelines;
	plan2linecost = costperlineplan2 * numberofphonelines;
	plan3linecost = costperlineplan3 * numberofphonelines;
	plan4linecost = costperlineplan4 * numberofphonelines;
	// multiplies costperline and numberofphonelines to create planxlinecost
	plan1GBcost = costperGBplan1 * numberofGBofdata;
	plan2GBcost = costperGBplan2 * numberofGBofdata;
	plan3GBcost = costperGBplan3 * numberofGBofdata;
	plan4GBcost = costperGBplan4 * numberofGBofdata; // will just evaluate to zero
	// multiplies costperGBplanx and numberofGBofdata to create planxGBcost
	plan1cost = plan1linecost + plan1GBcost;
	plan2cost = plan2linecost + plan2GBcost;
	plan3cost = plan3linecost + plan3GBcost;
	plan4cost = plan4linecost + plan4GBcost;
	// adds planxlinecost and planxGBcost to create the final cost of the plan "planxcost"
	cout << "\n\n\n";
	// creates some space
	cout << fixed << setprecision(2);
	// sets the decimal precision to two decimal points
	cout << setw(20) << left << "Plan 1 cost is:" << setw(5) << right << plan1cost << endl;
	cout << setw(20) << left << "Plan 2 cost is:" << setw(5) << right << plan2cost << endl;
	cout << setw(20) << left << "Plan 3 cost is:" << setw(5) << right << plan3cost << endl;
	cout << setw(20) << left << "Plan 4 cost is:" << setw(5) << right << plan4cost << endl;
	// prints out the cost for each plan
	return 0;
	// exit value of zero
}




