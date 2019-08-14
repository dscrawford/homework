// Lab 10 Exercise 1
// An array of 10 integers
//
// Program by: Daniel Crawford
#include <iostream>
using namespace std;

void getValue(int arrayvalues[]);
void displayarray(int value[]);

int main() {
    int value[10];
    getValue(value); // Receives the value then puts it into the array
    displayarray(value); // Displays the values in reverse order
    return 0;
}

void getValue(int arrayvalues[]) {
    int value;
    for (int number = 1; number <= 10; number++) {
        cout << "Enter number " << number << ":"; //Takes in the value
        cin >> value;
        arrayvalues[number] = value; //Puts the value into the array
    }

}

void displayarray(int value[]) {
    cout << "The values in reverse order are: " << endl;
    for (int number = 10; number >= 1; number--) {
        cout << "Value number " << number << " is " << value[number] << endl;
    }
    //Displays values in reverse order
}