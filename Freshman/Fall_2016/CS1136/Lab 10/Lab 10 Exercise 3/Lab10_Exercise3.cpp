// Lab 10 Exercise 3
// Fun with strings.
//
// Program by: Daniel Crawford
#include <iostream>
using namespace std;
//Prototype functions
int getnames(string nameslist[], int NAME_LIST_AMOUNT);
void displaynames(string nameslist[], int count);
//

int main() {
    const int NAME_LIST_AMOUNT = 10; // Size of the array is 10
    int count;
    string nameslist[NAME_LIST_AMOUNT];
    //nameslist is a 10 element large array
    count = getnames(nameslist, NAME_LIST_AMOUNT); //Gets the names and returns the amount of names
    cout << endl;
    displaynames(nameslist, count); //Displays all the names accordingly
    return 0;
}
int getnames(string nameslist[], int NAME_LIST_AMOUNT) {
    int count = 0;
    for(int n = 1; n <= 8; n++) {
        cout << "Enter name #" << n << ": ";
        cin >> nameslist[n];
        count++;
    }
    //Logic adds count up by one and recalls
    return count;
}
void displaynames(string nameslist[], int count) {
    for(int n = 1; n <= count; n++) {
        cout << "Name " << n << " is " << nameslist[n] << endl;
    }
    // For loop counts and prints out the array aligned with that number, as many times as the count went.
}