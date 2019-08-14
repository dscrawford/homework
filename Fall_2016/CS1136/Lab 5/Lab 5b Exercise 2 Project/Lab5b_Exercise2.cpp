// Lab 5b Exercise 2
// Call by value and call by reference
//
// Program by:    Daniel Crawford

#include <iostream>

using namespace std;

// function prototype
// Swap the values value1 and value2
void swapInts(int& value1, int& value2); //added an & after int in order to make it a reference
void process();

// process function â€“ does the work for the application
void process()
{
    int num1, num2;

    // ask for the first input value
    cout << "Enter number 1: ";
    cin >> num1;


    // get num2
    cout << "Enter number 2: ";
    cin >> num2;

    // OK, we now swap the values
    cout << "Swapping numbers 1 (" << num1 << ") and 2 ("
          << num2 <<")." << endl;

    swapInts(num1, num2);

    cout << "The new value of number 1 is " << num1
          << "." << endl;
    cout << "The new value of number 2 is " << num2
          << "." << endl;

}

// main function - you application starts here
int main()
{
    process();
    process();

    cout << "Thank you for using this application." << endl;

    return 0;
}

// This function is supposed to swap the values value1 and value2,
// but it will not work because value1 and value2 are passed by value.
void swapInts(int& value1, int& value2) //added an & after int in order to make it a reference
{
    int temp;
    // Swap value1 and value2
    temp = value1;
    value1 = value2;
    value2 = temp;
}