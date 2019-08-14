// Created by Daniel Crawford(dsc160130) on 9/17/2017

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;

// Program will Welcome and give product options to the customer, then the
// customer and choose what they want to do and add products to their cart.
public class Catalog_Section {
    public static void run(ArrayList<CatalogItem> bookscatalog,
                           ArrayList<CatalogItem>  dvdscatalog ) {
        int choice; //Variable to decide which option the user chooses.
        System.out.println("**Welcome to the Books and DVDs Store**" +
                        "(Catalog Section)\n");
        do { //Enter into the store, display options for user

            //Find a valid number between 1 and 6 or 9 and display options
            do {
                DisplayOptions();
                choice = Validator.getInt("Enter your choice: ");
                if ( (choice <= 0 || choice > 7) && choice != 9) {
                    System.out.println("This option is not acceptable");
                }
            } while ( (choice <= 0 || choice > 7) && choice != 9);
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
                    removeCatalogItem(bookscatalog, choice, "ISBN");
                    break;
                case 5:
                    //Remove DVD
                    removeCatalogItem(dvdscatalog, choice, "dvdcode");
                    break;
                case 6:
                    //Display Catalog
                    displayCatalogs(bookscatalog, dvdscatalog);
                    break;
                case 7:
                    //Check if there is actually data to backup, then continue.
                    if (bookscatalog.isEmpty() && dvdscatalog.isEmpty())
                        System.out.println("There are no dvds or books!");
                    else {
                        //Create a data with all underscores, formatted in
                        // yyyy_MM_dd_HH_mm_ss
                        Date date = new Date();
                        String formatted_date;
                        SimpleDateFormat formatstring =
                                new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
                        formatted_date = "catalog_backup_" +
                                formatstring.format(date);

                        UseFile.createFile(bookscatalog, dvdscatalog,
                                formatted_date);
                    }
                    break;
                case 9:
                    //Exit the store
                    System.out.println("Exiting the Catalog Section...\n");
                    break;
                default:
                    //A surprise message for any person who makes it here.
                    System.out.println("How did you get here");
                    break;
            }
            System.out.println();
        } while (choice != 9);
        //Sort both arraylists.
        Collections.sort(bookscatalog);
        Collections.sort(dvdscatalog);

    }

    private static void DisplayOptions() { //Displays after before each input
        System.out.print("Choose from the following options:\n"+
                "1 - Add Book\n" +
                "2 - Add AudioBook\n" +
                "3 - Add DVD\n" +
                "4 - Remove Book\n" +
                "5 - Remove DVD\n" +
                "6 - Display Catalog\n" +
                "7 - Create backup file\n" +
                "9 - Exit Catalog Section\n");
    }

    private static void AddCatalogItem(ArrayList<CatalogItem> catalogItems,
                                      int choice) {
        //title will be passed to new catalogitem object
        String title = Validator.getString("Enter the title: ");
        double price = Validator.getDouble("Enter the price: ");

        if (choice == 1 || choice == 2) { //If it is a book or audiobook
            int ISBN;
            do { //Find a valid and unique dvd code
                ISBN = Validator.getInt("Enter the ISBN: ");
            } while (!hasUniqueID(catalogItems, ISBN));
            //ask for and receive author
            String author = Validator.getString("Enter the author: ");

            if (choice == 2) { //If it is an audiobook
                 //Ask for runtime
                double runningtime = Validator.getDouble("Enter the runtime:" +
                        " ");
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
                dvdcode = Validator.getInt("Enter the dvdcode: ");
            } while (!hasUniqueID(catalogItems, dvdcode));

            String director = Validator.getString("Enter the director: ");
            int year = Validator.getInt("Enter the year it was released: ");

            catalogItems.add(new DVD(title,price,director,year,dvdcode));
        }
    }

    private static void removeCatalogItem(ArrayList<CatalogItem> catalogItems,
                                          int choice, String type) {
        //Remove a catalog item by ISBN or dvdcode
        if (!catalogItems.isEmpty()) {
            System.out.println(type + ":");
            displayCatalog(catalogItems);

            int ID = Validator.getInt("Enter the " + type + " to be removed:" +
                    " ");
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
        }
        else {
            System.out.println("The catalog is empty, cannot remove");
        }

    }

    private static void displayCatalogs(ArrayList<CatalogItem> BookItems,
                                       ArrayList<CatalogItem> DVDItems) {
        //Displays all the contents of the Book and DVD arrays
        System.out.println("Books:");
        if (!BookItems.isEmpty()) {
            displayCatalog(BookItems);
        }
        else { //if the book array is empty
            System.out.println("There are no books in the catalog.");
        }
        System.out.println("DVDs:");
        if (!DVDItems.isEmpty()) {
            displayCatalog(DVDItems);
        }
        else { // if the dvd array is empty
            System.out.println("There are no DVDs in the catalog.");
        }
    }

    private static void displayCatalog(ArrayList<CatalogItem> arr) {
        for (CatalogItem i : arr) {
            System.out.println(i.toString());
        }
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
}
