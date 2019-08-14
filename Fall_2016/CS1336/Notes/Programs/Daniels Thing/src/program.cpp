/*
 * program.cpp
 *
 *  Created on: Oct 10, 2016
 *      Author: dsc160130
 */
#include <iostream>
using namespace std;
int curve(int grade){
	for(grade >= 69; grade--);
	cout << grade << endl;
}
void processGrade(int grade)
		{
	if (grade >= 90)
	{
		cout << grade << ", LOL IM CURVING U DOWN\n";
		processGrade(grade - (grade - 69));
	}
	else if (grade >= 70)
	{
		cout << "hahahah you got a friggin " << grade << " not even an A u suk";
	}
	else
	{
		cout << grade << ", You have failed.";
	}
}
int main() {
	int grade;
	cout << "What is your grade?" << endl;
	cin >> grade;

	processGrade(grade);

	unsigned long long float nerd;
	nerd = 2.122141451234124123123125;
	cout << endl << sizeof(nerd) << endl;
	return 0;
}
