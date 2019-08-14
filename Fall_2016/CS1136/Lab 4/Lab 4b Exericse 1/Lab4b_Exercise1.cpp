// Lab 4b Exercise 1
// The compute_sum function
//
// Program by: Daniel Crawford
#include <iostream>
//std::cout, std::cin
using namespace std;
// Compute the sum of all of the numbers from 1 to n where n
// is a natural number
// use the formula: n(n+1)/2
int compute_sum(int limit) // compute_sum function -- Because I made this an int, now I can return a value
{
    int sum_to_limit;
    sum_to_limit = limit * (limit + 1) / 2;
    return sum_to_limit; //Returns whatever sum_to_limit is equal to
}
int main()
{
    int sum = 0; //sum is an unused integer literal at 0
    int maxNumber;
    // get the maxNumber for the function call
    cout << "Enter a whole number greater than 0" << endl;
    cin >> maxNumber;
    // call compute sum
    // If I left compute_sum(maxNumber) right here the computer would just calculate it then never use it again
    // display the sum calculated by the compute_sum function
    cout << "The sum of 1 to " << maxNumber;
    cout << " is " << compute_sum(maxNumber); // Moved compute_sum inside of the cout function.

    return 0;
}