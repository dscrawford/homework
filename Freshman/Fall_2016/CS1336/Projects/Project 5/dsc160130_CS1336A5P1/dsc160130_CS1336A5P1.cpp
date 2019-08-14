#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

using namespace std;

// Function prototypes
int readFile(int studentId[], double grades[], int maxGrades);
double minimumGrade(const double grades[], int numberGrades);
void displayForStudent(int id, const int studentId[],
                       const double grades[], int numberGrades);
// additional function prototypes go here

double maximumGrade(const double grades[], int numberGrades);
double averageGrade(const double grades[], int numberGrades);
void displayReverse(const int studentId[], double grades[], int numberGrades);

//

int main()
{
    // result returned from main
    int result;
    // create the arrays. Maximum number of grades supported is 20
    // You need to create a const int with a size of 20. Name it
    // MAX_GRADES The student ids need to be in an array named
    // studentId and the grades need to be in an array named
    // grades. Both of these need to be of size MAX_GRADES. The
    // student ids are of type int and the grades are of type
    // double.
    // insert the definitions here:
    const int MAX_GRADES = 20;
    int studentId[MAX_GRADES] = {0};
    double grades[MAX_GRADES] = {0};
    // this will contain the actual number of grades read in from the input file
    int numberGrades;
    // go read the file and fill in the student id and grades array.
    // the return value is the actual number read in.
    numberGrades = readFile(studentId, grades, MAX_GRADES);

    if (numberGrades == 0)
    {
        // output a message and return with a value of 4.
        // there were no grade records
        cout << "There are no grade records\n";
        result = 4;
    }
    else
    {
        // display the number of grade records read in
        cout << "There are " << numberGrades << " grade records\n";

        // display the contents of the arrays in reverse order
        // OPTIONAL - comment out the next statement out if you are
        // not doing the optional part.
        displayReverse(studentId, grades, numberGrades);

        // output numbers in the format: xx.xx
        cout << fixed << setprecision(2);

        // calculate and display the minimum grade
        cout << "Minimum grade is " <<
             minimumGrade(grades, numberGrades) << endl;

        // calculate and display the maximum grade
        // OPTIONAL - comment the cout statement out if you are
        // not doing the optional part.
        cout << "Maximum grade is " <<
             maximumGrade(grades, numberGrades) << endl;

        // calculate and display the average grade
        // OPTIONAL - comment the cout statement out if you are
        // not doing the optional part.
        cout << "Average grade is " <<
             averageGrade(grades, numberGrades) << endl;

        // for student ids 1 through 6 display the grades and average
        //  for that student
        for (int id=1; id<=6; id++)
        {
            cout << endl;
            // call displayForStudent to display grades and average for this student
            displayForStudent(id, studentId, grades, numberGrades);
        }

        // return value - processing worked without errors
        result = 0;
    }

    // return to operating system
    return result;
}

int readFile(int studentId[], double grades[], int maxGrades) {
    const string FILE_NAME = "grades.txt";

    ifstream inputFile;
    inputFile.open(FILE_NAME);
    if (inputFile.fail()) {
        cout << "The input file " << FILE_NAME << " failed to open." << endl;
        return 0;
    }
    //Program opens input file grades.txt
    //if it failed then it will return it back to the main function
    //Then the program will end

    int index = 1; //Reads from the first line
    while (inputFile >> studentId[index] >> grades[index]) {
        if (inputFile.eof() || index == 20) {
            break;
        }
        else {
            index++;
        }
    }
    // Reads input file from lines 1 - 20 only, or until the end of the file.
    inputFile.close();
    return index;
}

void displayReverse(const int studentId[], double grades[], int numberGrades) {
    cout << setw(10) << right << "Student Id" << setw(10) << right <<  "Grade" << endl;
    //Displays what each column means
    for (int n = numberGrades; n >= 1; n--) {
        cout << setw(10) << right << studentId[n]
             << setw(10) << right  << fixed << setprecision(2) <<  grades[n] << endl;
    }
    //Loop will display the studentId then their grade in reverse order of the file.
}

double maximumGrade(const double grades[], int numberGrades) {
    double maxGrade = -1;
    for (int n = 1; n <= numberGrades; n++) {
        if ( maxGrade < grades[n]) {
            maxGrade = grades[n];
        }
        else if (n == 1){
            maxGrade = grades[n];
        }
    }
    //Loop sets maxGrade equal to the first grade
    //Then checks to see if there is a number bigger than it, assigns it to that number if it is.
    return maxGrade;
}

double minimumGrade(const double grades[], int numberGrades) {
    double minGrade = -1;
    for (int n = 1; n <= numberGrades; n++) {
        if ( minGrade > grades[n]) {
            minGrade = grades[n];
        }
        else if (n == 1){
            minGrade = grades[n];
        }
    }
    //Loop sets minGrade equal to the first grade
    //Then checks to see if there is a number smaller than it, assigns it to that number if it is.
    return minGrade;
}

void displayForStudent(int id, const int studentId[], const double grades[], int numberGrades) {
    int count = 0;
    double avg = 0;
    cout << "Grades for student " << id << endl;
    for (int n = 1; n <= 20; n++) {
        if (studentId[n] == id) {
            cout << fixed << setprecision(2) << grades[n] << endl;
            count++;
            avg += grades[n];
        }
    }
    //Loop displays the grades for the student, increments count by 1 for how many grades the student id has
    count == 0 ? cout << "There are no grades for this student" :
    cout << "Average for student " << id << " is " << (avg / count) << endl;
    //If the count is 0, it will say there are no grades for the student
    //If the count is not 0, it will display the average for the student
}

double averageGrade(const double grades[], int numberGrades) {     //Calculates the average of all the grades in the file
    double avgGrade = 0;
    for (int n = 1; n <= numberGrades; n++) {
        avgGrade += grades[n];
        if (n == numberGrades) {
            avgGrade /= n;
        }
    }
    //Loop grabs all or first 20 grades and adds them all together
    //Once loop is finished, assigns avgGrade to the division of ( All the grades ) / ( How many grades)
    return avgGrade;
}