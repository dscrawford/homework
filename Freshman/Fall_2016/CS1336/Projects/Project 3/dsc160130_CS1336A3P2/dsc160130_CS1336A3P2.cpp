//Units sold to revenue program
//
//
//Program by Daniel Crawford

#include <iostream>
#include <limits>
#include <iomanip>

using namespace std;
//Prototype-functions-
int getQuantity();
double getDiscount(int quantity);
bool isPreferred(int quantity);
double calculateSale(int quantity, double discount);
void displaySale(int quantity, double discount, double revenue);
void displayPreferred();
int main();
// -------------------

int getQuantity() { //Obtains a quantity, tells user their input is invalid if they give bad input
    int itemssold;
    cout << "Enter the number of units sold: ";
    cin >> itemssold; //Asks for input of units sold
    while(1) {
        if (cin.fail() || itemssold < 0) {
            cout << "The number of units must be 1 or more\n";
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(),'\n');
            cout << "Enter the number of units sold: ";
            cin >> itemssold; //If input does not work, the program will tell the user then they can put in correct input
        }
        else {
            break; //Ends the while loop
        }
    }
    return itemssold; //Returns the cin value back to the main
}
double getDiscount(int quantity) { //Determines the discount the user will receive
    double discount;
    if(quantity >= 20 && quantity <= 39) {
        discount = 0.06;
    }
    else if(quantity >= 40 && quantity <= 64) {
        discount = 0.11;
    }
    else if(quantity >= 65 && quantity <= 199) {
        discount = 0.18;
    }
    else if(quantity >= 200 && quantity <= 299){
        discount = 0.25;
    }
    else if(quantity >= 300) {
        discount = 0.33;
    }
    else {
        discount = 0.00;
    }
    //If statements assign discount to the appropriate value for discount based on quantities value
    return discount;
}
bool isPreferred(int quantity) {
    const int PREFERRED_CUSTOMER_LEVEL = 270; //If they meet this number, then the customer is a preferred customer
    if(quantity >= PREFERRED_CUSTOMER_LEVEL) {
        return true; //returns true if quantity is greater than or equal to the customer level
    }
    else {
        return false; //returns false if the customer isn't.
    }
}
double calculateSale(int quantity, double discount) { //Calculates the total revenue the company will receive.
    const double RETAIL_PRICE = 109.45;//Retail price is the price of the unit
    double sale = (quantity * RETAIL_PRICE);
    sale = sale * (1.0 - discount); //Calculates sale/Revenue
    return sale; //Returns the value back to the main
}
void displaySale(int quantity, double discount, double revenue) {
    cout << "The customer purchased " << quantity << " units"; //Prints out how much the customer purchased
    if (discount == 0){ //If the customer did not get a discount, it will inform them
        cout << " and does not qualify for a discount\n";
    }
    else { //If the customer qualified for a discount, then they will be told how much of a discount they got
        cout << " and qualified for a " << static_cast<int>(discount * 100) << "% discount" << endl;
    }
    cout << "The total sale for the customer is $" << fixed << setprecision(2) << revenue << endl; //Displays the total revenue/sale received
}
void displayPreferred() {
    cout << "The customer is a preferred customer\n"; //Displays the customer is prefferred.
}

int main() {
    int numberofitems;
    double discount, revenue; //Defining variables
    numberofitems = -1; //Assign numberofitems to -1 to start the while loop
    while (numberofitems != 0) {
        numberofitems = getQuantity(); // Assigns numberofitems to whatever getQuantity returns
        if(numberofitems > 0) {
            discount = getDiscount(numberofitems); //Assigns discount to whatever getDiscount returns
            revenue = calculateSale(numberofitems, discount); //Assigns revenue to whatever calculateSale returns
            if (isPreferred(numberofitems)) { //If isPreferred is true, then this will run
                displaySale(numberofitems, discount, revenue);
                displayPreferred();
            }
            else { //If isPreferred is false, then this will run
                displaySale(numberofitems, discount, revenue);
            }
        }
    }
    return 0;// Exit value of 0
}