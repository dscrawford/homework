// Lab 7b Exercise 1
// Financial aid program

// Program by:  Daniel Crawford
#include <iostream>
// std::cout, std::cin

using namespace std;
//Prototype functions
char isundergrad(char student);
int financialaidamount(int income);
void displayfinancialaid(int financialaid);
//
char isundergrad(char student) {
    cout << "Are you an undergraduate student?";
    cin >> student; // Asks if the student is or isn't an undergraduate student
    if (student == 'y' || student == 'Y') {
        return 'T'; //Returns T is the user inputs y or Y
    }
    else if(student == 'n' || student == 'N') {
        return 'F'; // Returns F is the user inputs n or N
    }
    else {
        cout << "Invalid Input";
        return 'I'; // Returns I if the user inputs a bad value
    }
}
int financialaidamount(int income) {
    cout << "What is your yearly income?";
    cin >> income; // Asks for user to input their yearly income
    if (income > 0) {
        if (income <= 15000) {
            return 1000; //returns 1000 back to the main if it is lesser than or equal to 15000
        }
        else {
            return 500; //returns 500 if income is greater than 150000
        }
    }
    else {
        cout << "Financial aid could not be calculated";
        return -1; //returns -1 as an error value if the value is negative
    }
}
void displayfinancialaid(int financialaid){
    cout << "You qualify for $" << financialaid << " in financial aid.";
    //Prints out the amount of financial aid the student qualifies for
}


int main() {
    int income;
    char student; //defining variables income and student
    student = isundergrad(student); //assigns student either T or F
                                    //isundergrad() checks if the student is an undergraduate or not
    if(student == 'T'){ //T if the student is an undergrad
        income = financialaidamount(income); //assigns income to whatever amount of financial aid the user will get from it
        if (income >= 0){ //if income is a number above 0(will only be either 500 or 1000)
            displayfinancialaid(income); //display the financial aid
            return 0;
        }
        else { //If the input was invalid, or under 0
            return -1;
        }
    }
    else if(student == 'F') { //F if the student is not an undergrad
        cout << "You do not qualify for financial aid"; //Tells the user they do not qualify
        return -1;
    }
    else { //Anything else, the function returns an error
        return -1;
    }
    return 0;
}