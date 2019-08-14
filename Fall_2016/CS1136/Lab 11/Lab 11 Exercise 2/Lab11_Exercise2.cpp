// Lab 11 Exercise 2
// Sales by quarter
//
// Program by: Daniel Crawford
#include <iostream>
#include <iomanip>

using namespace std;

void getnames(string salesmen[], int number); //Gets the names of each salesman
void getquarter(string salesmen[], double quartersales[][4], int number); //Gets how much each salesman sold each quarter
void displayquarters(double quartersales[][4], string salesmen[], int number); //Shows who had the highest quarter sales

int main() {

    const int AMOUNT_OF_SALESMEN = 3;

    string salesmen[AMOUNT_OF_SALESMEN];
    double quartersales[AMOUNT_OF_SALESMEN][4];

    for (int n = 0; n <= 2; n++) {
        getnames(salesmen, n);
        getquarter(salesmen, quartersales, n);
        cout << endl;
    }
    //Loop goes through each salesmen and assigns their name to an array
    //Then it has a parallel array which holds the quartersales for that salesman

    cout << endl;

    for (int n = 0; n <= 3; n++) {
        displayquarters(quartersales, salesmen, n);
    }
    //Loop goes through each quarter and displays which salesman sells the most.

    return 0;
}

void getnames(string salesmen[], int number) {
    cout << "Enter in the name for salesman " << number + 1 << ": ";
    cin >> salesmen[number];
    //Asks for and receives input for salesman #(number)
}

void getquarter(string salesmen[], double quartersales[][4], int number) {
    cout << "Now enter in sales for each quarter for " << salesmen[number] << endl;
    for (int n = 0; n <= 3; n++) {
        cout << "Enter in data for quarter " << n + 1 << ": ";
        cin >> quartersales[number][n];
    }
    //Asks for and receives input for quarter 1 - 4 for a salesman
}

void displayquarters(double quartersales[][4], string salesmen[], int number) {
    double maxquarter = quartersales[0][number];
    int maxquarterP = 0;
    //Above assignments assigns the max value equal to the first salesman and their sale for that quarter
    for (int n = 1; n <= 2; n++) {
        if (quartersales[n][number] > maxquarter) {
            maxquarter = quartersales[n][number];
            maxquarterP = n;
        }
    }
    //Loop decides if there is a value above the initial maxquarter, if there is, then it reassigns it
    //to another salesman
    cout << "Salesman " << salesmen[maxquarterP] << " had the highest sales for quarter " << number + 1
         << " with $" << fixed << setprecision(2) << maxquarter << endl;
    //Displays who sold the most for each quarter
}