// Lab 3 Exercise 2
// Calculate MPH (Miles Per Hour) and KPH (Kilometers Per Hour).
//
// Program by: Daniel Crawford
#include <iostream>
#include <iomanip>
// grants new commands, iomanip in particular grants setw, setprecision and fixed
using namespace std;

int main() {
    float startmile, endmile, hours, totalmile, KMeterCalc;
    // defining variables as a float
    KMeterCalc = 1.60934;
    // making KmeterCalc(Kilometer Calculation) as a literal to convert miles to kilometers
    cout << "How much was your mileage before the trip? \n";
    cin >> startmile;
    // asks for input of mileage before the trip
    cout << "How much is your mileage after the trip? \n";
    cin >> endmile;
    // asks for input of mileage after the trip
    totalmile = (endmile - startmile);
    // assigns totalmile to final mileage minus starting mileage
    cout << "How long was the trip? \n";
    cin >> hours;
    // asks for input of how many hours the trip took
    cout << fixed << setprecision(1);
    // sets the decimal precision to the first decimal place
    cout << setw(30) << left << "Total Miles" << setw(10) << right << totalmile << endl;
    // calculates how many miles the trip was then prints
    cout << setw(30) << left << "Miles per hour:" << setw(10) << right << (totalmile)/hours << endl;
    // calculates miles per hour then prints
    cout << setw(30) << left << "Kilometers" << setw(10) << right << totalmile*KMeterCalc << endl;
    // calculates kilometers then prints
    cout << setw(30) << left << "Kilometers per hour:" << setw(10) << right << (totalmile*KMeterCalc)/hours << endl;
    // calculates kilometers per hour then prints
    return 0;
    // exit value of 0

}