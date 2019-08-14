// Created by Daniel Crawford(dsc160130) on 9/17/2017
import java.util.Scanner;
import java.util.ArrayList;
// Program will Welcome and give product options to the customer, then the
// customer and choose what they want to do and add products to their cart.
public class Crawford_Daniel_A2 {
    public static void main(String args[]) {
        String books[] = {"Intro to Java", "Intro to C++", "Python", "Perl",
                "C#"};
        String dvds[] = {"Snow white", "Cinderella", "Dumbo", "Bambi",
                "Frozen"};
        //Both used for display and transfering to the cart
        double booksPrices[] = {45.99, 89.34, 100.00, 25.00, 49.99};
        double dvdsPrices[]  = {19.99, 24.99,  17.99, 21.99, 24.99};
        // Used for both display, transfering to cart and calculations (Total
        // cost including tax)
        Scanner input = new Scanner(System.in);
        int choice; //Variable to decide which option the user chooses.
        ArrayList<String> CartProducts = new ArrayList<>();
        ArrayList<Double> CartPrices   = new ArrayList<>();
        //Both arraylists will be parallel and display prices and the
        // products when added to input.
        System.out.println("**Welcome to the Comets Books and DVDs Store**\n");
        do { //Enter into the store, display options for user
            do { //Find a valid number between 1 and 8 and display options
                DisplayOptions();
                choice = inputValidation();
                if (choice <= 0 || choice > 8) {
                    System.out.println("This option is not acceptable");
                }
            } while (choice <= 0 || choice > 8);
            switch (choice) {
                case 1:
                    displayArrays(books, booksPrices, "Books");
                    //Displays all the books in the store
                    break;
                case 2:
                    displayArrays(dvds, dvdsPrices, "Dvds"); //Displays
                    //  all the dvds in the store
                    break;
                case 3:
                    getInventoryNumber(books, booksPrices, CartProducts,
                            CartPrices, "Books"); //Gets the inventory
                    // number for a book, then adds it to the cart.
                    break;
                case 4:
                    getInventoryNumber(dvds, dvdsPrices, CartProducts,
                            CartPrices, "Dvds"); //Gets the inventory
                    // number for a dvd, then adds it to the cart
                    break;
                case 5:
                    displayArrays(CartProducts, CartPrices); //Displays all
                    // the items in the cart and their total with tax
                    break;
                case 6:
                    System.out.printf("Total: %.2f\n", getTotal(CartPrices));
                    clearArrays(CartProducts, CartPrices);
                    //Displays the current total cost
                    break;
                case 7:
                    clearArrays(CartProducts,CartPrices); //Emptys out the
                    // contents of the cart
                    break;
                case 8:
                    break;
            }
        } while (choice != 8);
        input.close();
    }
    private static void DisplayOptions() { //Displays after before each input
        System.out.print("Choose from the following options:\n"+
                "1 - Browse books inventory (price low to high)\n" +
                "2 - Browse DVDs inventory (price low to high)\n" +
                "3 - Add a book to the cart\n" +
                "4 - Add a DVD to the cart\n" +
                "5 - View cart\n" +
                "6 - Checkout\n" +
                "7 - Cancel Order\n" +
                "8 - Exit Store\n");
    }
    private static void displayArrays(String[] itemsArray, double[]
            pricesArray, String itemType) { //Displays all of a certain
        // product and their prices in the store.
        System.out.printf("%-20s%-22s%-6s", "Inventory Number", itemType,
                "Prices\n");
        System.out.println("-------------------------------------------------");
        int[] iarr = new int[pricesArray.length];
        for (int i = 0; i < pricesArray.length; ++i)
            iarr[i] = i + 1;
        double sortedprices[] = pricesArray.clone();
        String sorteditems[]  = itemsArray.clone();
        sort(sortedprices, 0, pricesArray.length - 1, iarr, sorteditems);
        for (int i = 0; i < pricesArray.length; ++i) {
            System.out.printf("%-20s%-22s$%.2f\n", iarr[i],
                    sorteditems[i], sortedprices[i]);
        }
        System.out.println("-------------------------------------------------");
    }
    private static void displayArrays(ArrayList<String> itemlist,
                                      ArrayList<Double> pricelist) {
        //Displays the produts and prices in the cart then the Total with tax
        if (itemlist.isEmpty()) {
            System.out.println("Your cart is empty.");
        }
        else {
            System.out.printf("%-18s%-6s", "Items", "Prices\n");
            System.out.println("------------------------");
            for (int i = 0; i < itemlist.size(); ++i) { //Displays all products
                // and prices
                System.out.printf("%-18s$%.2f\n", itemlist.get(i),
                        pricelist.get(i));
            }
            System.out.println("------------------------");
            System.out.printf("Total + Tax = %.2f\n", getTotal(pricelist));
            //Displays the total plus tax
        }
    }
    private static void getInventoryNumber(String[] itemsArray, double[]
            pricesArray, ArrayList<String> itemlist, ArrayList<Double>
            pricelist, String itemType) {
        if (itemlist.size() == 5) { //Limited arraylist to 5 elements.
            System.out.println("You've maxed out your cart!");
        }
        else {

            int choice;
            do { //Loop will continue until a choice between 1 and 5 is input
                System.out.println("Please enter the inventory number you " +
                        "want  to order. (or -1 if you want to redisplay the " +
                        " menu and 0 if you want to don't want to order)");
                choice = inputValidation();
                if ((choice <= 0 || choice > 5) && choice != -1) {
                    System.out.println("Please enter a value between 1 and 5 " +
                            "(or -1 if you want to redisplay the menu)");
                }
                else if (choice == -1) {
                    displayArrays(itemsArray, pricesArray, itemType);
                }
            } while ( (choice <= 0 || choice > 5));
            itemlist.add(itemsArray[choice - 1]);
            pricelist.add(pricesArray[choice - 1]);
        }
    }
    private static double getTotal(ArrayList<Double> pricelist) { //Will
        // iterate through array and total it
        double Total = 0;
        for (double i: pricelist) { //Loop through array and find total
            Total += i;
        }
        return Total + (Total * 0.0825);
    }
    private static void clearArrays(ArrayList<String> itemlist,
                                    ArrayList<Double> pricelist) {
        //Function clears both arrays.
        itemlist.clear();
        pricelist.clear();
    }
    private static int inputValidation() { //Function returns a validated int
        Scanner input = new Scanner(System.in);
        int value;
        //Value to be returned as validated input
        while (!input.hasNextInt()) { //Keep looping till an int is put in
            if (!input.hasNextInt()) {
                System.out.println("Please enter a valid number.");
            }
            System.out.print("Enter your choice: ");
            input.next();
        }
        value = input.nextInt();
        return value;
    }
    private static void partition(double arr[], int low, int high, int iarr[],
                                 String itemArr[]) { //Will swap each value
        // that is lesser than the pivot
        double pivot = arr[high]; //Set pivot to the current high
        int i = (low - 1);
        for (int j = low; j < high; j++) { //Loop from the lower index to the
            // higher index
            if (arr[j] <= pivot) { //If the index is letter than the pivot
                i++;
                double temp = arr[i]; //Swap arrays with the current low
                // index with each value higher than it in the pivot.
                arr[i] = arr[j];
                arr[j] = temp;

                int itemp = iarr[i]; //Swap the index array
                iarr[i] = iarr[j];
                iarr[j] = itemp;

                String itemtemp = itemArr[i]; //Swap the items array
                itemArr[i] = itemArr[j];
                itemArr[j] = itemtemp;
            }
        }
        double temp = arr[i + 1]; //Swap the price array
        arr[i+1] = arr[high];
        arr[high] = temp;

        int itemp = iarr[i + 1]; //Swap the index array
        iarr[i + 1] = iarr[high];
        iarr[high] = itemp;

        String itemtemp = itemArr[i + 1]; //Swap the items array
        itemArr[i + 1] = itemArr[high];
        itemArr[high] = itemtemp;
    }
    private static void sort(double arr[], int low, int high, int iarr[],
                             String itemArr[]) {
        //Quicksort implementation
        if (low < high) { //Run the function as long as this is true.
            partition(arr, low, high, iarr, itemArr);
            sort(arr, low, high - 1, iarr, itemArr); //Will decrement
            // until it reaches low == high
            sort(arr, low + 1, high, iarr, itemArr); //Will increment until
            // it reaches low == high
        }
    }
}
