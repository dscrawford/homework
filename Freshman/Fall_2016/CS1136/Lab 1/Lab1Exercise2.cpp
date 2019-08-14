/*
 * Lesson1Exercise2.cpp
 *
 *  Created on: Sep 10, 2016
 *      Author: Daniel-Mac
 */
// CS 1136 Lab 1 Exercise 2
// A program to calculate the user's age
// in dog years.
//
// Program by: Daniel Crawford

#include <iostream>
using namespace std;

int main()
{
	int humanAge, dogAge;
	cout << "How old are you? ";
	cin >> humanAge;
	dogAge = humanAge * 7;
	cout << "Wow, in dog years that is " << dogAge << ".\n";
	// UPDATE THE FOLLOWING CODE TO DISPLAY YOUR NAME
	cout << "Hello. My name is Daniel Crawford\n";
	return 0;
}




