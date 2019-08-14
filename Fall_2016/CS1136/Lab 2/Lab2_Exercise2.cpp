/*
 * Lab2Exercise2.cpp
 *
 *  Created on: Sep 11, 2016
 *      Author: Daniel-Mac
 */
// CS 1136 Lab 2 Exercise 2
// Program by: Daniel Crawford
#include <iostream>
// need iostream to compile
using namespace std;

int main() {
	float MAmount, MPrice;
	// three floats variables, respectively called Markup Amount and Manufacture Price
	double MPercentage = 0.425;
	// Markup Percentage is a literal, assigned to 45%
	cout << "What is the Manufacture Price? \n";
	cin >> MPrice;
	// Asks for MPrice then user will input it
	MAmount = MPrice * MPercentage;
	// Multiplies Mprice and Mpercentage after the user assigns MPrice
	cout << "The markup for the circuit board is: " << MAmount << endl;
	// Prints text then the Markup Amount
	cout << "The selling price for the circuit board is: " << MAmount + MPrice;
	// Afterwards it prints texts then computates MAmount plus MPrice
	return 0;
	// exit value of 0
}



