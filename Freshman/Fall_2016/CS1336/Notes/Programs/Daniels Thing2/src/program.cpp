/*
 * program.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: dsc160130
 */

#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstdlib>

const double PI = 3.14;

using namespace std;

double readValue()
{
	double value;
	cin >> value;

	if (value <= 0.0)
	{
		//cout << "The value is not valid and must be > 0. GO AWAY" << endl;
		//exit(EXIT_FAILURE);
		cout << "Please input a valid number above 0" << endl;
		return readValue();
	}
	return value;
}

void displayMenu()
{
	cout << "Choose one of the options 1 or 2" << endl;
	cout << "1. Calculate the area of a rectangle" << endl;
	cout << "2. Calculate the area of a circle" << endl;
}

bool execute()
{
	bool result;
	int menuItem;

	cin >> menuItem;
	if (menuItem == 1)
	{
		double length, width;
		cout << "enter the length: ";
		length = readValue();
		cout << "enter the width: ";
		width = readValue();
		cout << "(" << width << ")" << "*" << "(" << length << ")" << endl;
		cout << "the area is " << (width * length) << endl;
		result = true;
	}
	else if (menuItem == 2)
	{
		double radius;
		cout << "Enter the radius of the circle";
		radius = readValue();
		cout << "(" << PI << ")" << "*" << "(" << radius << ")" << "*" << "(" << radius << ")" << endl;
		cout << "The area is " << PI * radius * radius << endl;

		result = true;
	}
	else
	{
		cout << "The menu value of " << menuItem << " is not valid, please enter in either 1 or 2" << endl;
		result = execute();
	}
	return result;
	}

int main() {
	displayMenu();
	if (execute())
	{
		cout << "The program worked\n";
	}
	else
		cout << "The program did not work\n";
	return 0;
}

