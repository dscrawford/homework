// Created by Daniel Crawford on 8/30/2017
// Diana Nyad swim speeds and statistics
import java.sql.Time;
import java.util.Scanner;
//Program will gather data through two arrays containing the Distance and Time
//Then it will output the average speed, distance and year in a table
//All arrays will only hold shorts, program will ask you questions, please
//answer
public class Crawford_Daniel_A1 {

    public static void main(String[] args) {
	    short DistanceSwam[] = new short[5]; // Total miles swam each swim
	    short TimeToFinish[] = new short[5]; // Hours taken to finish a swim
	    short[] Year = new short[] {1974, 1975, 1978, 1979, 2013}; //for Display
        double avgSpeed = 0; //Will gather sum of Distance/Time then divide by 5
        Scanner input = new Scanner(System.in);
        for (int i = 0; i < 5; ++i) { //Gathers input for each year
            short temp; // Will gather input and be used for validation
            do { // Keep asking for distance swam until a valid number is input
                System.out.println("Enter the distance Diana Nyad swam in "
                        + Year[i]);
                temp = input.nextShort();
                if (temp > 0)
                    DistanceSwam[i] = temp;
                else
                    System.out.println("Please enter a valid short above 0.");
            } while (temp <= 0);
            temp = 0; //Temp won't leak into next input
            do { // Keep asking for hours took until a valid number is input
                System.out.println("How many hours did it take Nyad to swim" +
                        " the distance?");
                temp = input.nextShort();
                if (temp > 0)
                    TimeToFinish[i] = temp;
                else
                    System.out.println("Please enter a valid short above 0.");
            } while (temp <= 0);
            avgSpeed += (double)DistanceSwam[i]/TimeToFinish[i];
            //avgspeed = 0 + all speeds each year
        }
        avgSpeed /= 5;
        // Currently has all speeds from each year summed,
        // find average over 5 years
        System.out.print("+--------------------------------------------------" +
                         "---------------+\n" + "|Year | Distance (miles) | "  +
                         "Time (hours) | Speed (miles/hour) |\n" + "+--------" +
                         "---------------------------------------------------" +
                         "------+\n");
        for (int i = 0; i < 5; ++i) {
            //Loop will display all Data gathered formatted to fit above columns
            System.out.printf(" %-5s| %-17s| %-13s| %-19s%n", Year[i],
                            DistanceSwam[i] + " miles", TimeToFinish[i] +
                            " hours", (double)DistanceSwam[i]/TimeToFinish[i]);
        }
        System.out.println("\nDiana Nyads average speed is: "
                           + avgSpeed + " miles/hour"); //Display average speed
    }
}
