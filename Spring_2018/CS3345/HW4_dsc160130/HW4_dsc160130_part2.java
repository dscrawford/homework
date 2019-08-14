//Made by Daniel Crawford on 3/3/2018(dsc160130)
import java.io.BufferedReader;
import java.io.FileReader;

public class HW4_dsc160130_part2 {
    public static void main(String args[]) {
        RandomBT quicktree = new RandomBT();
        quicktree = constructTreeFromFile("books.txt", quicktree);

        System.out.println("Displayed in order: \n");
        quicktree.inorder();
        System.out.println("\nProblems in tree: \n");
        quicktree.inorderFindErrors();
    }
    public static RandomBT constructTreeFromFile(String file, RandomBT tree) {
        try {
            //Will try to open the file within the same directory as
            // HW4_dsc160130
            String PATH = HW4_dsc160130_part2.class.getProtectionDomain()
                    .getCodeSource().getLocation().getPath() + file;
            BufferedReader buffer = new BufferedReader(new FileReader(PATH));
            String parser;

            //Read through the file and store all values into AVL tree
            while ( (parser = buffer.readLine()) != null) {
                String[] strarr = parser.split(",");
                // strarr[0] = bookname, str[1] = author, str[2] = isbn
                Book book = new Book(strarr[0], strarr[1]);
                String ISBN = strarr[2];
                tree.insert(book, strarr[2]);
            }
        }
        catch (java.io.IOException e) {
            System.out.println("ERROR: " + e.getMessage());
        }
        return tree;
    }
}
