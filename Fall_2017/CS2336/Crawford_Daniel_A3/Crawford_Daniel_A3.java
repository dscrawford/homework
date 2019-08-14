// Created by Daniel Crawford(dsc160130) on 9/17/2017
import java.util.ArrayList;
import java.util.Scanner;
// Program will Welcome and give product options to the customer, then the
// customer and choose what they want to do and add products to their cart.
public class Crawford_Daniel_A3 {
    public static void main(String args[]) {
        //Catalog of books
        ArrayList<CatalogItem> bookscatalog = new ArrayList<>();
        //Catalog of dvds
        ArrayList<CatalogItem>  dvdscatalog  = new ArrayList<>();
        int choice; //Variable to decide which option the user chooses.
        System.out.println("**Welcome to the Books and DVDs Store**" +
                        "(Catalog Section)\n");
        do { //Enter into the store, display options for user
            do { //Find a valid number between 1 and 8 and display options
                DisplayOptions();
                choice = intValidation();
                if ( (choice <= 0 || choice > 6) && choice != 9) {
                    System.out.println("This option is not acceptable");
                }
            } while ( (choice <= 0 || choice > 6) && choice != 9);
            switch (choice) {
                case 1:
                    //Add Book
                    AddCatalogItem(bookscatalog, choice);
                    break;
                case 2:
                    //Add AudioBook
                    AddCatalogItem(bookscatalog, choice);
                    break;
                case 3:
                    //Add DVD
                    AddCatalogItem(dvdscatalog, choice);
                    break;
                case 4:
                    //Remove Book
                    System.out.print("Enter the ISBN you want to delete: ");
                    if(removeCatalogItem(bookscatalog, choice)) //If the
                        // removal is successful, display the catalog.
                        displayCatalog(bookscatalog,dvdscatalog);
                    break;
                case 5:
                    //Remove DVD
                    System.out.print("Enter the dvdcode you want to delete: ");
                    if(removeCatalogItem(dvdscatalog, choice)) //If the
                        // removal is successful, display the catalog.
                        displayCatalog(bookscatalog,dvdscatalog);
                    break;
                case 6:
                    //Display Catalog
                    displayCatalog(bookscatalog, dvdscatalog);
                    break;
                case 9:
                    //Exit the store
                    System.out.println("Exiting the store...");
                    break;
                default:
                    //A surprise message for any person who makes it here.
                    System.out.println("How did you get here");
                    break;
            }
        } while (choice != 9);
    }
    private static void DisplayOptions() { //Displays after before each input
        System.out.print("Choose from the following options:\n"+
                "1 - Add Book\n" +
                "2 - Add AudioBook\n" +
                "3 - Add DVD\n" +
                "4 - Remove Book\n" +
                "5 - Remove DVD\n" +
                "6 - Display Catalog\n" +
                "9 - Exit Store\n");
    }
    private static void AddCatalogItem(ArrayList<CatalogItem> catalogItems,
                                      int choice) {
        System.out.print("Enter the title: ");
        String title = StringValidation(); //title will be passed to new catalogitem object

        double price = 0; //price will be passed to new catalogitem object
        do {
            System.out.print("Enter the price: ");
            Scanner input = new Scanner(System.in);
            double value;
            price = doubleValidation();
            if (price <= 0) {
                System.out.println("This input is not acceptable");
            }
        } while (price <= 0);
        if(choice == 1 || choice == 2) { //If it is a book or audiobook
            int ISBN;
            do { //Find a valid and unique dvd code
                System.out.print("Enter the ISBN: ");
                ISBN = intValidation();
                if(!hasUniqueID(catalogItems,ISBN)) {
                    System.out.println("The ISBN you entered is already in " +
                            "use.");
                }
            } while (!hasUniqueID(catalogItems, ISBN));

            System.out.print("Enter the author: "); //ask for and receive author
            String author = StringValidation();

            if (choice == 2) { //If it is an audiobook
                 //Ask for runtime
                System.out.print("Enter the runtime: ");
                double runningtime = doubleValidation();
                catalogItems.add(new AudioBook(title,price,author, ISBN,
                        runningtime));
            }
            else { //If it is a book
                catalogItems.add(new Book(title,price,author, ISBN));
            }
        }
        else { //If it is a DVD
            int dvdcode;
            do { //Find a valid and unique dvd code
                System.out.print("Enter the dvdcode: ");
                dvdcode = intValidation();
                if(!hasUniqueID(catalogItems,dvdcode)) {
                    System.out.println("The Dvdcode you entered is already in" +
                            " use.");
                }
            } while (!hasUniqueID(catalogItems, dvdcode));

            System.out.print("Enter the director: ");
            String director = StringValidation();

            System.out.print("Enter the year it was released: ");
            int year = -1;
            while (year < 0) {
                year = intValidation();
                if (year < 0) {
                    System.out.println("Please enter a valid year.");
                }
            }

            catalogItems.add(new DVD(title,price,director,year,dvdcode));
        }
    }
    private static boolean removeCatalogItem(ArrayList<CatalogItem>
                                                  catalogItems, int choice) {
        //Remove a catalog item by ISBN or dvdcode, if none exists return false
        int ID = intValidation();
        boolean foundID = false;
        String CatalogType;
        if(choice == 4)
            CatalogType = "Book";
        else
            CatalogType = "DVD";
        for(int i = 0; i < catalogItems.size(); ++i) {
            //Find the catalog item and remove it
            if (catalogItems.get(i).getID() == ID) {
                catalogItems.remove(i);
                foundID = true;
            }
        }
        if (!foundID)
            System.out.println("The " + CatalogType + " doesn't exist in the " +
                    "Catalog");
        return foundID;

    }
    private static void displayCatalog(ArrayList<CatalogItem> BookItems,
                                       ArrayList<CatalogItem> DVDItems) {
        //Displays all the contents of the Book and DVD arrays
        System.out.printf("--------------------------------------------" +
                "-------------------------------------------------------" +
                "--------------\n" + "Book information\n" +
                "--------------------------------------------------------" +
                "---------------------------------------------------------\n");
        if (!BookItems.isEmpty()) {
            for (CatalogItem i : BookItems) {
                System.out.println(i.toString());
            }
        }
        else { //if the book array is empty
            System.out.println("There are no books in the catalog.");
        }
        System.out.printf("--------------------------------------------" +
                "-------------------------------------------------------" +
                "--------------\n" + "DVD information\n" +
                "--------------------------------------------------------" +
                "---------------------------------------------------------\n");
        if (!DVDItems.isEmpty()) {
            for (CatalogItem i: DVDItems) {
                System.out.println(i.toString());
            }
        }
        else { // if the dvd array is empty
            System.out.println("There are no DVDs in the catalog.");
        }
        System.out.println();
    }
    private static boolean hasUniqueID(ArrayList<CatalogItem>
                                                     catalogItems, int ID) {
        //Determine uniqueness of an ID(ISBN or dvdcode)
        boolean isUniqueID = true;
        for(CatalogItem i: catalogItems) { //Check each ID to see if it
            // already exists
            if (i.getID() == ID) {
                isUniqueID = false;
                break;
            }
        }
        return isUniqueID;
    }
    private static int intValidation() {
        //Retrieve a valid int
        Scanner input = new Scanner(System.in);
        int value;
        while (!input.hasNextInt()) {
            input.nextLine();
            if (!input.hasNextInt()) {
                System.out.println("This option is not acceptable");
            }
        }
        value = input.nextInt();
        return value;
    }
    private static String StringValidation() {
        //Retrieve a non-empty String
        Scanner input = new Scanner(System.in);
        String value = "";
        while (value.isEmpty()) {
            value = input.nextLine();
            if(value.isEmpty()) {
                System.out.println("This input is not acceptable, please " +
                        "re-enter.");
            }
        }
        return value;
    }
    private static double doubleValidation() {
        //Retrieve a valid double.
        Scanner input = new Scanner(System.in);
        double value;
        while (!input.hasNextDouble() ) {
            input.nextLine();
            if (!input.hasNextDouble()) {
                System.out.println("This input is not acceptable\n");
            }
        }
        value = input.nextDouble();
        return value;
    }
}
