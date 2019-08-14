#include <iostream>
//std::cout, std::cin
using namespace std;
//don't have to use std:: anymore
int main() {
    unsigned int length1, width1, length2, width2;
    unsigned int rect1, rect2;
    //Defining the variables first and second of each length and width and each rectangle
    cout << "Enter the length then the width of the first rectangle" << endl;
    cin >> length1 >> width1;
    //Asks for the length and widgth of the first triangle
    cout << "Enter the length then the width of the second rectangle" << endl;
    cin >> length2 >> width2;
    //Asks for the length and width of the second triangle
    rect1 = length1 * width1;
    rect2 = length2 * width2;
    //Calculates both rectangles area.
    if(rect1 > rect2)
        cout << "The first rectangle is larger!";
    else if(rect2 > rect1)
        cout << "The second rectangle is larger!";
    else
        cout << "They are both equal.";
    //If statements make it so the program will tell you which rectangle is larger.
    return 0;
    //exit value of 0
}