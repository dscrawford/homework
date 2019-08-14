import java.util.ArrayList;

public class Crawford_Daniel_TermProject {
    public static void main(String[] args) {
        System.out.println("**Welcome to the Comets Books and DVDs Store**\n");
        ArrayList<CatalogItem> bookscatalog = new ArrayList<>();
        ArrayList<CatalogItem>  dvdscatalog = new ArrayList<>();
        String choice;

        do {
            displayoptions();
            choice = Validator.getString("Enter your choice: ");
            switch(choice) {
                case "A":
                    String username = Validator.getString("Please enter your" +
                            " username: ");
                    String password = Validator.getString("Please enter your" +
                            " password: ");
                    if (UseFile.login(username,password))
                        Catalog_Section.run(bookscatalog, dvdscatalog);
                    else
                        System.out.println("Invalid username or password.\n");
                    break;
                case "B":
                    Customer_Section.run(bookscatalog, dvdscatalog);
                    break;
                case "C":
                    System.out.println("Exiting the store...");
                    break;
                default:
                    System.out.println("\"" + choice + "\" is not a valid " +
                            "choice.");
            }
        } while (!choice.equals("C"));
    }

    private static void displayoptions() {
        System.out.print("Please select your role:\n" + "A - store manager\n" +
                "B - customer\n" + "C - exit store\n");
    }
}
