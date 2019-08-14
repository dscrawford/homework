package AlgPack;

public class ListCreator {
    //Create an array that has all elements in order(least to most)
    public static int[] inOrder(int size) {
        int[] arr = new int[size];
        //Loop and add in order array(from least to most)
        for (int i = 0; i < size; i++) {
            arr[i] = i;
        }

        return arr;
    }

    //Create an array that has all of its elements in reverse order(most to
    // least)
    public static int[] reverseOrder(int size) {
        int[] arr = new int[size];

        for (int i = size, j = 0; j < size; j++) {
            arr[j] = i;
            i--;
        }

        return arr;
    }

    //Create an array that is half sorted, and half unsorted(will be
    // generated using Math.Random
    public static int[] almostOrder(int size, int MaxValue) {
        int[] arr = new int[size];

        for (int i = 0; i <= size/2; i++) {
            arr[i] = i;
        }

        for (int i = size/2 + 1; i < size; i++) {
            arr[i] = (int) (Math.random() * MaxValue);
        }

        return arr;
    }


    //Create an array that is completely random, unique values are not
    // guaranteed in this version.
    public static int[] randomOrder(int size, int MaxValue) {
        int[] arr = new int[size];

        for (int i = 0; i < size; i++) {
            arr[i] = (int) (Math.random() * MaxValue);
        }

        return arr;
    }

}
