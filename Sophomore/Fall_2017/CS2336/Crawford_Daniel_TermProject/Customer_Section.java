// Created by Daniel Crawford(dsc160130) on 9/17/2017
import java.util.Iterator;
import java.util.Scanner;
import java.util.ArrayList;
// Program will Welcome and give product options to the customer, then the
// customer and choose what they want to do and add products to their cart.
public class Customer_Section {
    public static void run(ArrayList<CatalogItem> bookscatalog,
                           ArrayList<CatalogItem>  dvdscatalog) {
        Scanner input = new Scanner(System.in);
        int choice;

        ArrayList<CatalogItem> BookCart = new ArrayList<>();
        ArrayList<CatalogItem> DvdCart = new ArrayList<>();

        System.out.println("**Welcome to the Comets Books and DVDs Store**\n\n");
        do {
            boolean hasAdded = false, hasDeleted = false; //For adding and
            // deleting
            int ISBN, dvdcode; //For adding/deleting


            DisplayOptions();
            //Finds a valid choice between 1 and 9
            do {
                choice = Validator.getInt("Enter your choice:");
                if (choice <= 0 || choice > 9) {
                    System.out.println("Please enter a value between 1 and 9");
                }
            } while (choice <= 0 || choice > 9);
            switch (choice) {
                case 1:
                    //displays all the books in the catalog
                    displayArrays(bookscatalog, "Books");
                    break;

                case 2:
                    //displays all the dvds in the catalog
                    displayArrays(dvdscatalog, "Dvds");
                    break;

                case 3:
                    //Tries to add a book by ISBN to the cart, if it doesn't
                    // exist it will warn the user.
                    hasAdded = false;
                    if (!bookscatalog.isEmpty()) {
                        displayArrays(bookscatalog, "Books");
                        ISBN = Validator.getInt("Enter the ISBN of the " +
                                "book: ");
                        hasAdded = addCatalogItem(bookscatalog, BookCart,
                                ISBN);
                        if (!hasAdded)
                            System.out.println("That ISBN does not exist");
                    }
                    else
                        System.out.println("The books catalog is empty");

                    break;

                case 4:
                    //Tries to add a DVD by dvdcode, if it doesn't exist it
                    // will warn the user.
                    hasAdded = false;
                    if (!bookscatalog.isEmpty()) {
                        displayArrays(dvdscatalog, "Dvds");
                        dvdcode = Validator.getInt("Enter the dvdcode of the" +
                                " Dvd: ");
                        hasAdded = addCatalogItem(dvdscatalog, DvdCart,
                                dvdcode);
                        if (!hasAdded)
                            System.out.println("That dvdcode does not exist");
                    }
                    else
                        System.out.println("The dvds catalog is empty.");
                    break;

                case 5:
                    //Tries to delete a book by ISBN, if it can't find it it
                    // will tell the user.
                    if(!BookCart.isEmpty()) {
                        hasDeleted = false;
                        displayArrays(BookCart, "Books");
                        ISBN = Validator.getInt("Enter the ISBN of the " +
                                "book: ");
                        hasDeleted = deleteCatalogItem(BookCart, ISBN);
                        if (!hasDeleted)
                            System.out.println("That ISBN was not in the cart.");
                    }
                    else
                        System.out.println("The book cart is empty.");
                    break;

                case 6:
                    //Tries to delete the DVD by ISBN, if it can't find it it
                    // will tell the user.
                    if(!DvdCart.isEmpty()) {
                        hasDeleted = false;
                        displayArrays(DvdCart, "Dvds");
                        dvdcode = Validator.getInt("Enter the dvdcode of the" +
                                " Dvd: ");
                        hasDeleted = deleteCatalogItem(DvdCart, dvdcode);
                        if (!hasDeleted)
                            System.out.println("That dvdccode was not in the cart" +
                                    ".");
                    }
                    else
                        System.out.println("The dvd cart is empty");
                    break;

                case 7:
                    //Displays all arrays and the total with tax.
                    displayArrays(BookCart, "Books");
                    displayArrays(DvdCart, "Dvds");
                    System.out.printf("Your total(including tax): %.2f\n",
                            (getTotal(BookCart) + getTotal(DvdCart)) );
                    break;

                case 8:
                    //Displays, gives total including tax and clear the cart.
                    displayArrays(BookCart, "Books");
                    displayArrays(DvdCart, "Dvds");
                    System.out.printf("Your total(including tax): %.2f\n",
                            (getTotal(BookCart) + getTotal(DvdCart)) );
                    clearArray(BookCart);
                    clearArray(DvdCart);
                    break;

                case 9:
                    //A nice goodbye message to get those customers reeling
                    // back to us.
                    System.out.println("Thanks for shopping with us!");
                    break;
            }
            System.out.println();
        } while (choice != 9);
    }

    private static void DisplayOptions() { //Displays after before each input
        System.out.print("Choose from the following options:\n"+
                "1 - Browse books inventory (price low to high)\n" +
                "2 - Browse DVDs inventory (price low to high)\n" +
                "3 - Add a book to the cart\n" +
                "4 - Add a DVD to the cart\n" +
                "5 - Delete a book from cart\n" +
                "6 - Delete a DVD from cart\n" +
                "7 - View cart\n" +
                "8 - Checkout\n" +
                "9 - Done Shopping\n");
    }

    private static void displayArrays(ArrayList<CatalogItem> arr, String type) {
        //Displays one arraylist of catalogitems.
        System.out.println(type + ":");
        if (!arr.isEmpty()) {
            for (CatalogItem i: arr)
                System.out.println(i.toString());
        }
        else
            System.out.println("This Catalog is empty.");
    }

    private static double getTotal(ArrayList<CatalogItem> arr) {
        //Totals the total amount including tax.
        double total = 0;
        if (!arr.isEmpty()) {
            total = 0;
            for (CatalogItem i: arr)
                total += i.getPrice();
            return total + (total * 0.0825);
        }
        return total;
    }

    private static void clearArray(ArrayList<CatalogItem> arr) {
        //clears an array.(only carts will be cleared)
        arr.clear();
    }

    private static boolean addCatalogItem(ArrayList<CatalogItem> catalog,
                                       ArrayList<CatalogItem> cart, int id) {
        //Searches for the id to add, if it finds a matching id then it will
        // add it to the cart and exit.
        boolean hasID = false;
        for (CatalogItem i: catalog) {
            if (i.getID() == id) {
                cart.add(i);
                hasID = true;
                break;
            }
        }
        return hasID;
    }

    private static boolean deleteCatalogItem(ArrayList<CatalogItem> cart, int id) {
        //Scans for a matching ID, if it finds one is marks it found and
        // deletes the item.
        boolean hasDeleted = false;
        Iterator<CatalogItem> iter = cart.iterator();

        //iterate through the array and find the equivalent id, if there are
        // products with the same id, it will only remove one of them.
        if (!cart.isEmpty()) {
            while (iter.hasNext()) {
                CatalogItem i = iter.next();
                if (i.getID() == id) {
                    iter.remove();
                    hasDeleted = true;
                    break;
                }
            }
        }
        return hasDeleted;
    }
}
