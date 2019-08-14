// Lab 5b Exercise 3
// Currency conversions 
//
// Program by:    Daniel Crawford
#include <iostream>
#include <iomanip>

using namespace std;

// This program will get an amount in US dollars and convert it 
// to another currency

// Prototypes of the functions
float convertToYen(float dollars);
void convertToEurosAndPesos(float dollars, float& Euros, float& Pesos);

int main ()
{  
    float dollars, euros, pesos, yen;
    cout << fixed << showpoint << setprecision(2);
    cout << "Please input the amount of US Dollars "
         << "you want converted: ";
    cin >> dollars;

    cout << "The value of " << "$" << dollars << " is:" << endl;
    yen = convertToYen(dollars);
    // Executes convertToYen and assigns yen to the returned value
    convertToEurosAndPesos(dollars, euros, pesos);
    // Executes convertToEurosAndPesos, euros and pesos are referenced value after they are calculated.
    cout << yen << " Yen" << endl;
    cout << euros << " Euros" << endl;
    cout << pesos << " Pesos" << endl;
    //Prints out the conversions

    //  Fill in the code to convert to yen, euros, and pesos
    // and display the results

    return 0;
}

float convertToYen(float dollars)
{
    float Yen, Yen_To_Dollar = 104.75; //sets the Yen_To_Dollar variable equal to its dollar to yen value
    Yen = dollars * Yen_To_Dollar; //Calculates Yen
    return Yen; //Returns back the value of Yen
}
void convertToEurosAndPesos(float dollars, float& Euros, float& Pesos) //float& changes the variables after they are modified
{
    float Dollars_To_Euros = 0.77, Dollars_To_Pesos = 13.07; //sets the conversions values from pesos and euros to dollars
    Euros = dollars * Dollars_To_Euros; //Calculates dollars to Euros
    Pesos = dollars * Dollars_To_Pesos; //Calculates dollars to Pesos
}