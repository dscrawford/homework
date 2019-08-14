package AlgPack;

import java.util.Arrays;

public class SortAlg {

    //AlgPack.SortAlg supplied by powerpoint
    //Worst case, n(n-1)/2 comparisons are used
    public static ReturnInformation InsertionSort(int [] arr) {

        ReturnInformation data = new ReturnInformation();

        for (int i=1; i < arr.length; i++) {
            //One comparison done here
            data.comparisons++;
            int temp = arr[i];
            int j;
            for (j = i - 1; j >= 0 && temp < arr[j]; j--) {
                data.comparisons += 2; //Two comparisons done here.
                arr[j + 1] = arr[j];
                data.movements++; //Movement done here.
            }
            arr[j + 1] = temp;

            data.movements++; //Movement done here, even if there is not
            // re-assignment, there is still time being taken.
        }

        return data;
    }

    //AlgPack.SortAlg supplied by powerpoint
    //n(n-1)/2 comparisons is worst case done here.
    public static ReturnInformation SelectionSort(int[] arr) {

        ReturnInformation data = new ReturnInformation();

        for (int i = 0; i < arr.length - 1; i++) {
            //One comparison done here
            data.comparisons++;
            int index = i;

            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] < arr[index])
                    index = j;
                //Two comparisons done per iteration
                data.comparisons += 2;
                }

            int smallerNumber = arr[index];
            arr[index] = arr[i];
            arr[i] = smallerNumber;
            //Swap, so two movements.
            data.movements += 2;
        }

        return data;
    }

    //Merge sort supplied by the powerpoint
    //Below code is for Merge Sort
    //Expected comparisons will be nlog(n).

    public static ReturnInformation MergeSort(int inputarr[]) {

        int arr[] = inputarr;
        int length = inputarr.length;
        int[] tempMergeArr = new int[length];
        ReturnInformation data = new ReturnInformation();

        doMergeSort(0, length - 1, tempMergeArr, arr, data);

        return data;

    }

    private static ReturnInformation doMergeSort(int lowerIndex, int higherIndex, int[]
            tempMergeArr, int[] arr, ReturnInformation data) {

        if (lowerIndex < higherIndex) {
            int middle = lowerIndex + (higherIndex - lowerIndex) / 2;
            data = doMergeSort(lowerIndex,middle,tempMergeArr,arr, data);
            data = doMergeSort(middle+1,higherIndex,tempMergeArr,arr, data);
            data = mergeParts(lowerIndex,middle,higherIndex,tempMergeArr,arr, data);
        }
        return data;
    }

    private static ReturnInformation mergeParts(int lowerIndex, int middle,
                                                int higherIndex, int[] tempMergeArr,
                                                int[] arr, ReturnInformation data) {
        for (int i = lowerIndex; i <= higherIndex; i++) {
            //One comparison done here
            data.comparisons++;
            tempMergeArr[i] = arr[i];
            //One movement to new array
        }
        int i = lowerIndex;
        int j = middle + 1;
        int k = lowerIndex;
        while (i <= middle && j <= higherIndex) {
            //Two comparisons done here
            data.comparisons += 2;
            if (tempMergeArr[i] <= tempMergeArr[j]) {
                //One comparison done here
                data.comparisons++;
                arr[k] = tempMergeArr[i];
                //One movement to new array
                data.movements++;
                i++;
            }
            else {
                //One comparison done to reach here
                data.comparisons++;
                arr[k] = tempMergeArr[j];
                data.movements++;
                j++;
            }
            k++;
        }
        while (i <= middle) {
            //One comparison per iteration
            data.comparisons++;
            arr[k] = tempMergeArr[i];
            //One movement per iteration
            data.movements++;
            k++;
            i++;
        }
        return data;
    }

    //Quick sort is supplied by geeksforgeeks (https://www.geeksforgeeks.org/quick-sort/)
    //Below code is all used for quicksort
    //Worst case is n^2

    public static ReturnInformation quickSort(int[] arr) {
        ReturnInformation[] data = {new ReturnInformation()};
        quickSort(arr, 0, arr.length - 1, data);
        return data[0];
    }

    private static void quickSort(int[] arr, int low, int high,
                                               ReturnInformation[] data) {
        if (low < high) {
            data[0].comparisons++;

            int pi = partition(arr, low, high, data);
            quickSort(arr, low, pi - 2, data);
            quickSort(arr, pi, high, data);
        }
    }
    private static int partition(int[] arr, int low, int high,
                                               ReturnInformation[] data) {
        int pivot = arr[high];
        int i = (low - 1);

        for (int j = low; j < high; j++) {
            data[0].comparisons++;
            if (arr[j] <= pivot) {
                data[0].comparisons++;
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                data[0].movements += 2;
            }
        }
        int temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;
        data[0].movements += 2;
        return i+1;
    }

    //Radix sort is supplied by geeksforgeeks (https://www.geeksforgeeks.org/radix-sort/)
    //Radix sort has time O(dn)
    //Generally, no comparison are done except for iterating through for loop.

    //Also i kinda brute forced so java would pass by reference so thats why i
    // am making a length 1 array.

    public static ReturnInformation radixSort(int arr[]) {
        ReturnInformation[] temp = {new ReturnInformation()};
        int m = getMax(arr, arr.length, temp);

        for (int exp = 1; m/exp > 0; exp *= 10) {
            countSort(arr,arr.length,exp, temp);
        }

        return temp[0];
    }
    public static int getMax(int arr[], int n, ReturnInformation[] data) {
        int max = arr[0];
        for (int i = 1; i < n; i++) {
            max = arr[i];
            data[0].movements++;
        }
        return  max;
    }

    static void  countSort(int arr[], int n, int exp, ReturnInformation[] data) {
        int output[] = new int[n];
        int i;
        int count[] = new int[10];
        Arrays.fill(count,0);


        for (i = 0; i < n; i++ ) {
            count[(arr[i]/exp)%10]++;
            //One comparison per iteration
            data[0].comparisons++;
        }

        for (i = 1; i < 10; i++) {
            count[i] += count[i-1];
            //One comparison per iteration
            data[0].comparisons++;
        }

        for (i = n - 1; i >= 0; i--) {
            output[count[(arr[i]/exp)%10] - 1] = arr[i];
            //One movement for each radix
            data[0].movements++;
            count[(arr[i]/exp)%10]--;
            //One comparison per iteration
            data[0].comparisons++;
        }

        for (i = 0; i < n; i++) {
            arr[i] = output[i];
            //One movement for each new value added
            data[0].movements++;
            //One comparison per iteration
            data[0].comparisons++;
        }
    }

    //Below code is provided by geeksforgeeks (www.geeksforgeeks.org/heap-sort)
    //Worst case-time of heap sore is O(logn)
    public static ReturnInformation heapSort(int arr[]) {
        ReturnInformation[] data = {new ReturnInformation()};
        int n = arr.length;

        for (int i = n / 2 - 1; i >= 0; i--) {
            //One comparison done per iteration
            data[0].comparisons++;

            heapify(arr, n, i, data);
        }

        for (int i = n - 1; i >= 0; i--) {
            data[0].comparisons++;
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;
            //Two movements done on swap
            data[0].movements += 2;
            heapify(arr, i, 0, data);
        }

        return data[0];
    }
    private static void heapify(int arr[], int n, int i,
                               ReturnInformation[] data) {
        int largest = i;
        int l = 2*i + 1;
        int r = 2*i + 2;

        if(l<n && arr[l] > arr[largest]) {
            largest = l;
        }
        //Two comparisons done each call
        data[0].comparisons += 2;

        if(r < n && arr[r] > arr[largest])
            largest = r;
        //Two comparisons done each call
        data[0].comparisons += 2;

        if (largest != i) {
            //One comparison per iteration
            data[0].comparisons++;
            int temp = arr[i];
            arr[i] = arr[largest];
            arr[largest] = temp;
            //Two movements done on swap
            data[0].movements += 2;
            heapify(arr,n,largest, data);
        }
    }

    //Use this for making sure arrays are ordered correctly.
    private static void displayArr(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.println(arr[i]);
        }
    }
}
