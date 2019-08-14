//Created by Daniel Crawford(dsc160130) on 11/21/2017
//This program will take in a non-empty string and then count the amount of
//occurences in each character there are(The weight) Afterwards, the program
// will combine the lowest weight characters in pairs until they are combined
// into a single string.

import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;
import java.util.ArrayList;
public class Crawford_Daniel_A4 {
    public static void main(String args[]) {
        String str;
        char[] charArray;
        //Will find how many times a character reoccurs
        HashMap<String,Integer> charCount = new HashMap<>();
        //Will hold all values in order of its weight.

        //Assigns to a non-empty string
        System.out.print("Please enter a line of text: ");
        str = validateString();
        //For traversing the string.
        charArray = str.toCharArray();
        TranformCharArrayToLowerCase(charArray);

        for (int i = 0; i < str.length(); i++) { //Put the values in the
            // string into a hashmap
            if (charCount.containsKey( String.valueOf(charArray[i])) ) {
                charCount.put(String.valueOf(charArray[i]),  charCount.get
                        (String.valueOf(charArray[i])) + 1);
            }
            else {
                charCount.put(String.valueOf(charArray[i]), 1);
            }
        }

        //Queue will order all characters in terms of their weight.
        ArrayList<String> PriorityQueue = new ArrayList<>();
        ArrayList<Integer> Weights = new ArrayList<>();
        createPriorityQueue(PriorityQueue, charCount, Weights);

        displayAllCharacters(charCount, PriorityQueue);

        do { //Loop will continue to combine the priority queue until it
            // becomes a single value.
            System.out.println("Please press enter to continue..");
            try {
                System.in.read();
            } catch (IOException e) {
                e.printStackTrace();
            }

            System.out.println();
            //Combines the nodes and displays which nodes were combined
            System.out.println(combineNodes(PriorityQueue, charCount, Weights));
            //Sorts the array back into a priority queue.
            PriorityQueueSort(PriorityQueue, Weights);
            displayAllCharacters(charCount, PriorityQueue);
        } while (PriorityQueue.size() != 1);
        System.out.println("The program is complete. Shutting down...");
    }

    private static String validateString() { //Will make sure a string
        Scanner input = new Scanner(System.in);
        String s = ""; //s will be returned.
        while (s.isEmpty()) {
            s = input.nextLine();
            if (s.isEmpty()) {
                System.out.print("Only enter a valid string.\n" +
                        "Please enter a line of text: ");
            }
        }
        return s;
    }

    private static void displayAllCharacters(HashMap<String,Integer> map,
                                            ArrayList<String> arr)
    { //Displays all the characters and their weight.
        System.out.printf("%-36s|%36s\n", "Character", "Weight");
        for (String i: arr) {
            System.out.printf("%-36s|%36s\n", i, map.get(i));
        }
    }

    private static void TranformCharArrayToLowerCase(char[] array) {
        //Transforms any uppercase characters to lowercase
        for (int i = 0; i < array.length; i++) {
            array[i] = Character.toLowerCase(array[i]);
        }
    }

    private static void createPriorityQueue(ArrayList<String> arr,
                                                        HashMap<String,Integer> map,
                                                        ArrayList<Integer> weights)
    {
        ArrayList<Integer> temp = new ArrayList<>();
        map.forEach( (key, value) -> { //Assigns each value inside the
            // hashmap to a array formed like a priority queue
            weights.add(value);
            arr.add(key);
        });
        PriorityQueueSort(arr, weights); //Sorts all the values to place the
        // string array from the highest weight to the lowest weight.
    }

    private static void PriorityQueueSort(ArrayList<String> arr,
                                         ArrayList<Integer> weights) {
        //Selection sort
        for (int i = 0; i < arr.size(); i++) {
            int min = weights.get(i);
            int swapindex = i;
            for (int j = i; j < arr.size(); j++) {
                if (weights.get(j) < min) {
                    min = weights.get(j);
                    swapindex = j;
                }
            }
            swap(arr, i, swapindex);
            swap(weights, i, swapindex);
        }

    }

    private static <E> void swap(ArrayList<E> arr, int i, int j) { //Swaps
        // elements in an arraylist
        E temp = arr.get(i);
        arr.set(i, arr.get(j));
        arr.set(j, temp);
    }

    private static String combineNodes(ArrayList<String> arr,
                                    HashMap<String,Integer> map,
                                      ArrayList<Integer> weights) {
    //Will combine the smallest nodes together. (The last 2 in the queue)
        //Tiebreaker of weights will be determined by how close they are
        // together in the priority queue.
        String combinedNode;
        String returnmessage;
        int newWeight = 0;
        //Will get the weight of the lowest of the queue.
        if (arr.size() > 1) { //Will combine two nodes, then remove the nodes
            // that were combined to be replaced by a combined node.
            newWeight = map.get(arr.get(0)) +
                    map.get(arr.get(1)); //Assign to first 2 indexes in
            // priority queue.
            combinedNode = arr.get(0) + arr.get(1);

            returnmessage = "Combined node {" + arr.get(0) + "} " +
                    "(weight: {" + map.get(arr.get(0) )+ "}) " +
                    "with node {" + arr.get(1) + "} " +
                    "(weight: {" + map.get(arr.get(1)) +  "}) to" +
                    " produce node {" + combinedNode + "} (weight {" +
                    newWeight + "})";

            map.remove(arr.get(1));
            map.remove(arr.get(0));
            arr.remove(0);
            weights.remove(0);
        }
        else {
            combinedNode = arr.get(0);
            newWeight = map.get(arr.get(0));
            returnmessage = "No combinations were done.";
        }
        //Sets the last node in the priority queue to the combined node
        arr.set(0, combinedNode);
        //Sets the last node in the array of weights to the new weight
        weights.set(0, newWeight);
        //Puts the new combined node and its corresponding weight in the hashmap
        map.put(combinedNode, newWeight);
        return returnmessage;
    }
}
