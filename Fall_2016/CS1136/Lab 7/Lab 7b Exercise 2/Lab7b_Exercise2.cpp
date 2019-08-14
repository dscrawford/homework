// Lab 7b Exercise 2
// Convert to Roman numerals
//
//Program by:   Daniel Crawford
#include <iostream>
using namespace std;
//Prototype functions
int asknumber(int number);
void romannumeral(int number);
//
int asknumber(){
    int number;
    cout << "Enter a number from 1 to 5: ";
    cin >> number;
    //Asks the user to input a number
    if(number >= 1 && number <= 5) { //If the number is between 1 and 5
        return number; //Returns back number
    }
    else { //If the number is not between 1 and 5
        cout << "The number must be in the range of 1 through 5 inclusive";
        return 0; //Returns back 0 as an error value
    }
}
void romannumeral(int number){ //Decides the roman numeral
    switch(number) {
        case 1: {
            cout << "The roman numeral is I";
            break;
        }
        case 2: {
            cout << "The roman numeral is II";
            break;
        }
        case 3: {
            cout << "The roman numeral is III";
            break;
        }
        case 4: {
            cout << "The roman numeral is IV";
            break;
        }
        case 5: {
            cout << "The roman numeral is V";
            break;
        }
        //Switch statement will decide what roman numeral the number will print out.
    }
}
int main() {
    int number = asknumber(); // Assigns number to the result of asknumber()
    if (number != 0){ //If the value is not equal to 0, it will continue.
        romannumeral(number);
    }
    //If the value is 0, the program will move on and end
    return 0;
}