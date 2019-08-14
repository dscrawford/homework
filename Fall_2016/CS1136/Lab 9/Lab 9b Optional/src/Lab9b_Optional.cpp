#include <iostream>
#include <iomanip>
#include <limits>

//Prototypes
void draw(int width);
int getinput();

using namespace std;

int main() {
	int width;
	width = getinput(); //Retrieves a valid input
	draw(width); // Draws the program
	return 0;

}

int getinput() {
	int width;
	do {
		cout << "Enter a width between 4 and 80: ";
		cin >> width;
		if (width < 4 && width > 80) {
			cout << "Invalid value. Value must be between 4 and 80\n";
		}
		else if (cin.fail()) {
			cin.clear();
			cin.ignore(std::numeric_limits<int>::max(),'\n');
			cout << "Invalid value. Value must be between 4 and 80\n";
			width = -1;
		}
	} while(width < 4 && width > 80);
	//Validates user input then returns the input
	return width;
}

void draw(int width) {
	for(int space = 0; space <= width; space++) { //For loop for how many times a * should be printed
		if(space == 0) {
			cout << "*\n"; //Starts out with only one *
		}
		else if(space < width && space != 0) {
			cout << "*" << setw(space + 1) << "*\n"; //Draws a star and then a star an incremental space from it each time
		}
		else {
			for(int n = 0; n <= space; n++) {
				cout << "*"; //Fills the bottom with stars
			}
		}
	}
}
