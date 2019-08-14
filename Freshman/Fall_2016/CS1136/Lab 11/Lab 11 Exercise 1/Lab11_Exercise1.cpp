// Lab 11 Exercise 1
// Sales for each salesman
//
// Program by: Daniel Crawford
#include <iostream>
#include <iomanip>

using namespace std;

void getnames(string salesmen[], int number); //Gets the names of each salesman
void getquarter(string salesmen[], double quartersales[][4], int number); //Gets how much each salesman sold each quarter
double totalsales(double quartersales[][4], int number); //Calculates the total sales

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

    for (int n = 0; n <= 2; n++) {
        cout << "Total sales for " << salesmen[n] << " is $" << fixed << setprecision(2)
             << totalsales(quartersales, n) << endl;
    }
    //Loop goes through eah salesman and displays their name and how much they sold.


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

double totalsales(double quartersales[][4], int number) {
    int calculation = 0;
    for (int n = 0; n <= 3; n++) {
        calculation += quartersales[number][n];
    }
    return calculation;
    //Calculates the total amount of sales for each salesman.
}