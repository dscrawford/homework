import java.io.*;
import java.util.ArrayList;

public class UseFile {
    public static boolean login(String username, String password) {
        boolean canlogin = false;
        String findUserName = "", findPassword = ""; //Program will find
        try {
            //Open a file in src called credentials.txt
            File file = new File("src/credentials.txt");
            FileReader fileReader = new FileReader(file);
            BufferedReader credentials =
                    new BufferedReader(fileReader);
            String bufferstring;

            //loop will go through all of credentials.txt line by line, and
            // determine if the corresponding username's and passwords are
            // correct.
            boolean foundUserName = false;
            while ( (bufferstring = credentials.readLine()) != null  &&
                    !foundUserName) {
                char[] temp = bufferstring.toCharArray();
                StringBuilder temp_s = new StringBuilder();

                //Iterate through the line and find the username and
                // password, set it to findUserName and findPassword
                for(char c: temp) {
                    //loop ignores spaces
                    if (c != ' ' && c != ',') {
                        temp_s.append(c);
                    }
                    //if there is a comma, then the password is to the right
                    // of it.
                    else if ( c == ',') {
                        findUserName = temp_s.toString();
                        temp_s = new StringBuilder();
                    }
                }
                //this determines if the username exists.(there can not be
                // duplicate usernames in the text file)
                if (findUserName.equals(username)) {
                    findPassword = temp_s.toString();
                    foundUserName = true;
                }
            }
            //validates that the password is correct(case sensitive)
            if (findPassword.equals(password) && foundUserName)
                canlogin = true;
            fileReader.close();
            credentials.close();
        }
        catch (IOException e) { //determines if file exists
            System.out.println("ERROR: " + e);
        }
        return canlogin;
    }

    public static void createFile(ArrayList<CatalogItem> books,
                                    ArrayList<CatalogItem> dvds, String name) {
        try
            //Create an output file to print line by line
            File outfile = new File(name + ".txt");
            FileOutputStream fileOutputStream = new FileOutputStream(outfile);
            BufferedWriter bufferedWriter =
                    new BufferedWriter(new OutputStreamWriter(fileOutputStream));

            //Print all the books in the arraylist
            bufferedWriter.write("Books:\n");
            for (int i = 0; i < books.size(); ++i) {
                bufferedWriter.write(books.get(i).toString() + "\n");
            }

            //Print all the dvds in the arraylist.
            bufferedWriter.write("Dvds:\n");
            for (int i = 0; i < dvds.size(); ++i) {
                bufferedWriter.write(dvds.get(i).toString() + "\n");
            }
            bufferedWriter.close();
            fileOutputStream.close();
        }
        catch (IOException e) {
            System.out.println("ERROR: " + e);
        }
    }

}
