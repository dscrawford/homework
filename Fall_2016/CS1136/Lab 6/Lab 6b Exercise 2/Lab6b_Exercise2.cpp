// Lab 6b Exercise 2
// History grading project
//
// Program by: Daniel Crawford
#include <iostream> //std::cout, //std::cin
#include <iomanip> //std::setw, fixed, right, left
using namespace std;

//Prototype Functions
void testinput(int& grade1, int& grade2, int& grade3); //Input test grades into here, will return back invalid if the test grades are too low or high
void displaytests(int grade1, int grade2, int grade3); // Displays the test grades
void letter_grade(int totalpoints,char& lettergrade); //Decides what the letter grade is
//
void testinput(int& grade1, int& grade2, int& grade3) { //Each variable is sent back to the main through reference
    cout << "Enter the score for test #1: ";
    cin >> grade1;
    cout << "Enter the score for test #2: ";
    cin >> grade2;
    cout << "Enter the score for test #3: ";
    cin >> grade3;
    //Takes in the input of each of the 3 test grades
    if ((grade1 < 0 || grade2 < 0 || grade3 < 0) || (grade1 > 50 || grade2 > 50 || grade3 > 50)) {
        cout << "Please enter a valid test grade between 0 and 50\n";
        testinput(grade1, grade2, grade3);
        //This condition checks to see if the test grades are valid(between 0 and 50, if it is not, then it resets
        // the function and tells the user to re-enter the input
    }
    cout << endl; //Creates some extra space
}
void displaytests(int grade1, int grade2, int grade3) { //Function just displays the values, nothing is changed
    cout << setw(15) << left << "First Test:" << setw(2) << right << grade1 << endl;
    cout << setw(15) << left << "Second Test:" << setw(2) << right << grade2 << endl;
    cout << setw(15) << left << "Third Test:" << setw(2) << right << grade3 << endl;
    //Displays the values of each test grade
}
void letter_grade(int totalpoints,char& lettergrade) {
    if(totalpoints >= 92){ // If the total points are greater than 92
        lettergrade = 'A';
    }
    else if((totalpoints < 92) && (totalpoints >= 82)) { // If the total points are between 92 and 82
        lettergrade = 'B';
    }
    else if ((totalpoints < 82) && (totalpoints >= 72)) { // If the total points are between 82 and 72
        lettergrade = 'C';
    }
    else { // If the total points are below 72
        lettergrade = 'F';
    }
    //This function grabs total points accumulated and decides what the users final letter grade is.
    //Assigns the variable lettergrade to a Char
}
int main() {
    int grade1, grade2, grade3; //defines test grade variables
    int totalpoints; //totalpoints after a test is dropped.
    char lettergrade; //is defined by what the totalpoints are.
    testinput(grade1, grade2, grade3);
    displaytests(grade1, grade2, grade3);
    if(grade1 > grade2) {
        totalpoints = grade1 + grade3;
        cout << "After dropping test #2, the points earned are " << totalpoints << endl;
        //If the first test grade is better, then it only adds tests 1 and 3
        //Prints out total points
    }
    else if(grade2 > grade1) {
        totalpoints = grade2 + grade3;
        cout << "After dropping test #1, the points earned are " << totalpoints << endl;
        //If the second test grade is better, then it only adds tests 2 and 3
        //Prints out total points
        }
    else {
        cout << "You did the same on Test 1 and Test 2.\n";
        totalpoints = grade1 + grade3;
        cout << "After dropping test #2, the points earned are " << totalpoints << endl;
        //If they are both the same, totalpoints is assigned to the first test plus the third test
        //Prints out total points
    }
    letter_grade(totalpoints,lettergrade); //Decides the letter grade
    cout << "The letter grade is " << lettergrade; //Prints out the letter grade.
    return 0;
}