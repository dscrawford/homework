#include <iostream>
// std::cout, std::cin
#include <iomanip>
// std::fixed, std::setprecision, std::setw, std::left, std::right
using namespace std; //don't have to use std::
const double examweight = 0.55; //global variable for the constant value of exams weight
const double projectweight = 0.45; //global variable for the constant value of projects weight
double calculateProjectGrade(double value1, double value2, double value3, double value4) { //value(1-4) are input from the processStudent()
    double average = (value1 + value2 + value3 + value4) / 4; //calculates the average of project grades
    return average; //returns average to processStudent
}
double calculateExamGrade(double value1, double value2, double value3) { //value(1-3) are input from the processStudent()
    double average = (value1 + value2 + value3) / 3; //calculates the average of exam grades
    return average; //returns average to processStudent
}
double calculateFinalGrade(double exam, double project) { //exam and project are the accumulative grade of each type
    double finalgrade = (examweight * exam) + (projectweight * project); //calculates the final grade
    return finalgrade; //returns the final grade
}
void displayGradeInformation(double exam, double project, double final) {
    cout << fixed << setprecision(1); //sets decimal precision to 1
    cout << setw(12) << left << "Type" << setw(10) << right << "Average" << setw(10) << right << "Weight" << endl;
    cout << setw(12) << left << "Projects" << setw(10) << right << project << setw(10) << right << static_cast<int>(projectweight*100) << endl;
    cout << setw(12) << left << "Exams" << setw(10) << right << exam << setw(10) << right << static_cast<int>(examweight*100) << endl << endl;
    cout << setw(12) << left << "Final Grade" << setw(10) << right << final << endl;
    //prints out all of the formatted text and prints out the exam, project and final grade

}
double processStudent() {
    double project1, project2, project3, project4;
    double exam1, exam2, exam3;
    double finalexam, finalproject, finalgrade;
    //assigning variables for the final grades and all the project and exam grades
    cout << "Enter in the 4 project grades: ";
    cin >> project1 >> project2 >> project3 >> project4;
    cout << "Enter in the 3 exam grades: ";
    cin >> exam1 >> exam2 >> exam3;
    finalexam = calculateExamGrade(exam1, exam2, exam3); //assigns finalexam to the result of the accumulative exam grade
    finalproject = calculateProjectGrade(project1, project2, project3, project4); //assigns finalproject to the accumulative project grade
    finalgrade = calculateFinalGrade(finalexam, finalproject); //assigns finalgrade to the average points at the end
    displayGradeInformation(finalexam, finalproject, finalgrade); //displays the information about the grade
    return finalgrade; //returns final grade to the main function
}
int main() {
    double student, student2, student3, student4; // defined the student grades as a double
    student = processStudent();
    student2 = processStudent();
    student3 = processStudent();
    student4 = processStudent();
    //after processstudent returns the value, each value will now be assigned to a specific student so that the average between them all can be calculated
    cout << "\nThe average of all students is: " << (student + student2 + student3+ student4) / 4;
    //prints out the average grade of all the students accumulatively
    return 0; //exit value of 0
}