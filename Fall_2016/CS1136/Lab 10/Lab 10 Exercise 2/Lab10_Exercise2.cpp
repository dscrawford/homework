// Lab 10 Exercise 2
// Still more counting
//
// Program by: Daniel Crawford
#include <iostream>
using namespace std;

//Prototype functions
void valuecounter(int valuelist[], int value);
void display(int valuelist[]);
//

int main() {
    int value, valuelist[10] = {0};
    //Initialize all the values in valuelist
    do {
        cout << "Enter a one-digit number or 10 to exit: ";
        cin >> value;

        //Takes value from user

        if (value < 0 || value > 10) {
            cout << "The value " << value << " is not valid\n";
            cin.clear();
            //Rejects an input that is not between 0 and 9
        }
        else if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<int>::max(), '\n');
            cout << "Invalid input.\n";
            //Will reject a bad input
        }
        else {
            valuecounter(valuelist, value);
            //increments the value of valuelist[number] by one
        }
    } while (value != 10);
    cout << endl;
    display(valuelist); // Displays

    return 0;
}

void valuecounter(int valuelist[], int value) {
    valuelist[value]++; //Increments the count in the array by one
}

void display(int valuelist[]) {
    for(int n = 0; n <= 9; n++) {
        if(valuelist[n] != 0) {
            cout << "You entered " << valuelist[n] << " " << n << "(s)" << endl;
        }
    }
    //Displays each value that is above 0 in count.
}