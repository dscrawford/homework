import java.util.Scanner;

public class Validator implements Acceptable{
    private static Scanner input = new Scanner(System.in);
    private static Validator check = new Validator();

    public boolean isNonEmptyString(String s) {
        return s.isEmpty();
    }
    public boolean isPositiveInput(double d) {
        return d >= 0;
    }
    public boolean isPositiveInput(int i) {
        return i >= 0;
    }

    public static String getString(String message) { //returns non-empty string
        String s;

        do {
            System.out.print(message);
            s = input.nextLine();
            if (s.isEmpty())
                System.out.println("Please enter a non-empty string");
        } while (check.isNonEmptyString(s));
        return s;
    }

    public static double getDouble(String message) { //returns positive double
        double d;
        do {
            System.out.print(message);
            while (!input.hasNextDouble()) {
                System.out.print(message);
                input.nextLine();
                if(!input.hasNextDouble())
                    System.out.println("Please ONLY enter a valid double.");
            }
            d = input.nextDouble();
            input.nextLine();
        } while (!check.isPositiveInput(d));
        return d;
    }

    public static int getInt(String message) { //returns an int
        int i;
        do {
            System.out.print(message);
            while (!input.hasNextInt()) {
                System.out.print(message);
                input.nextLine();
                if(!input.hasNextInt())
                    System.out.println("Please ONLY enter a valid double.");
            }
            i = input.nextInt();
            input.nextLine();
        } while (!check.isPositiveInput(i));
        return i;
    }
}
