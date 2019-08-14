// Lab 4b Exercise 2
// Determine the minimum and maximum target heart rates.
//
// Program by:     Daniel Crawford

#include <iostream> //std::cout, std::cin
#include <iomanip> //std::setw, std::fixed, std::left, std::right, std::setprecision
// What include do you need for the i/o manimulators?

using namespace std; // wont have to enter in std

int calculateMaximumHeartRate(int age) //function is an int so it can return a value
{
    int Max_Heart_Rate = 220 - age; // Maximum Heart rate
    return Max_Heart_Rate;
    //calculates the maximum heart rate here
}
// Your displayTargetHeartRate function goes here
void displayTargetHeartRate(int heartrate) //function is a void because it does not need to return a value
{
    double Min_Trgt_Heart_Rate = calculateMaximumHeartRate(heartrate) * 0.6;
    double Max_Trgt_Heart_Rate = calculateMaximumHeartRate(heartrate) * 0.7;
    //calculates the Maximum and Minimum Heart rate here
    cout << fixed << setprecision(1); // Sets decimal precision to 1
    cout << setw(20) << left << "Minimum Target Heart Rate is:" << setw(10) << right << Min_Trgt_Heart_Rate << endl;
    cout << setw(20) << left << "Maximum Target Heart Rate is:" << setw(10) << right << Max_Trgt_Heart_Rate << endl;
    //Displays the Maximum and Minimum Target Heart Rate

}

int processTargetHeartRate() //Asks for the age then runs displayTargetHeartRate
{
    int age; // Defines age as an integer
    cout << "Enter your age: ";
    cin >> age;
    // Asks for age then is input
    displayTargetHeartRate(age); // calls to displayTargetHeartRate() which uses calculateMaximumHeartRate() to
                                 // display the final result
}

int main() //main function
{
	processTargetHeartRate();
	processTargetHeartRate();
	processTargetHeartRate();
    // will ask for age then print it out 3 times before the program ends
	return 0; // exit value of 0
}