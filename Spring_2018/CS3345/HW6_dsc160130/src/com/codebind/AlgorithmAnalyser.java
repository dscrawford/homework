package com.codebind;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.DocumentListener;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.EventQueue;

import AlgPack.ListCreator;
import AlgPack.ReturnInformation;
import AlgPack.SortAlg;


public class AlgorithmAnalyser {
    class WinningAlgorithm {
        private String name;
        private long movementsAndComparisons;

        WinningAlgorithm() {
            this.name = "None so far";
            this.movementsAndComparisons = -1;
        }

        WinningAlgorithm(String name, long movementsAndComparisons) {
            this.name = name;
            this.movementsAndComparisons = movementsAndComparisons;
        }

        public String getName() {
            return this.name;
        }

        public long getMovementsAndComparisons() {
            return this.movementsAndComparisons;
        }
    }

    private JPanel panel1;
    private JButton insertionSortButton;
    private JButton selectionSortButton;
    private JButton quickSortButton;
    private JButton mergeSortButton;
    private JButton heapSortButton;
    private JButton radixSortButton;
    private JButton createTheListButton;
    private JRadioButton inOrderRadioButton;
    private JRadioButton reverseOrderRadioButton;
    private JRadioButton almostOrderRadioButton;
    private JRadioButton randomRadioButton;
    private JFormattedTextField WinningAlgorithm_formattedTextField;
    private JFormattedTextField experimentalResultsFormattedTextField;
    private JSlider arraySize_slider;
    private JTextField winningAlgorithm_textField;
    private JTextField sliderValue_textField;
    private JTextField DataType_textField;
    private JTextField SortType_textField;
    private JTextField Comparisons_textField;
    private JTextField Movements_textField;
    private JTextField totalTime_textfield;
    private JTextField size_textField;

    private int[] arr;
    private int sliderSize;
    private WinningAlgorithm winAlg;

    //After each sorting operation, set all the display values.
    private void displayInformation(ReturnInformation data, String sortType,
                                    long timetaken, int arrSize) {
        Comparisons_textField.setText( Long.toString(data.getComparisons()) );

        totalTime_textfield.setText( Long.toString(timetaken)+ " nanoseconds");

        DataType_textField.setText("Integer");

        size_textField.setText( Integer.toString(arrSize) );

        Movements_textField.setText( Long.toString(data.getMovements()) );

        SortType_textField.setText( sortType );
    }

    //Compare algorithms and compete.
    private WinningAlgorithm determineWinningAlgorithm(WinningAlgorithm winAlg,
                                           WinningAlgorithm competingAlg) {
        //If there is no winning algorithm so far
        if (winAlg == null) {
            winAlg = competingAlg;
            winningAlgorithm_textField.setText(competingAlg.getName());
        }

        //If the competing algorithm has better efficiency
        if (competingAlg.getMovementsAndComparisons() <
                winAlg.getMovementsAndComparisons()) {
            winAlg = competingAlg;
            winningAlgorithm_textField.setText(winAlg.getName());

        }

        return winAlg;
        //Otherwise make no changes, including for a tie.
    }

    private WinningAlgorithm resetWinningAlgorithm(WinningAlgorithm winAlg) {
        winAlg = null;
        winningAlgorithm_textField.setText("None so far");
        return winAlg;
    }

    public AlgorithmAnalyser() {
        //Buttongroup will make the buttons individual so they cannot be all
        // selected together.
        ButtonGroup buttonGroup = new ButtonGroup();
        buttonGroup.add(reverseOrderRadioButton);
        buttonGroup.add(inOrderRadioButton);
        buttonGroup.add(almostOrderRadioButton);
        buttonGroup.add(randomRadioButton);

        winningAlgorithm_textField.setText("None so far");

        arraySize_slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                //Slidersize will determine how big the array will be
                sliderSize = arraySize_slider.getValue();

                //Set the sliderValue to show the size;
                sliderValue_textField.setText(Integer.toString(sliderSize));
            }
        });

        createTheListButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //If the slider value has changed, reset the winning algorithm.
                if (arr != null) {
                    if (arr.length != sliderSize);
                        winAlg = resetWinningAlgorithm(winAlg);
                }

                arr = new int[sliderSize];
                //If reverse order is selected
                if (reverseOrderRadioButton.isSelected()) {
                    arr = ListCreator.reverseOrder(sliderSize);
                }
                //If inOrder is selected
                else if (inOrderRadioButton.isSelected()) {
                    arr = ListCreator.inOrder(sliderSize);
                }
                //If almostOrder is selected
                else if (almostOrderRadioButton.isSelected()) {
                    arr = ListCreator.almostOrder(sliderSize, 20000);
                }
                //If randomOrder is selected
                else if (randomRadioButton.isSelected()) {
                    arr = ListCreator.randomOrder(sliderSize, 20000);
                }
            }
        });

        /*
         * BELOW ARE ALL THE SORTING ALGORITHM BUTTONS
         */

        insertionSortButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (arr != null) {
                    //Keep track of how long it took to run the program and
                    // how many comparisons were done, how many movements.
                    long start_time, timetaken;

                    //Create temparr so the other array remains untouched.
                    int temparr[] = arr.clone();

                    start_time = System.nanoTime();
                    ReturnInformation data = SortAlg.InsertionSort(temparr);

                    //Calculate the time taken to complete insertion sort
                    timetaken = System.nanoTime() - start_time;

                    //Update the results
                    displayInformation(data, "Insertion", timetaken, temparr.length);

                    //Find new winning algorithm
                    WinningAlgorithm competingAlg = new WinningAlgorithm
                            ("Insertion", data.getComparisons()
                                    + data.getMovements());

                    winAlg = determineWinningAlgorithm(winAlg, competingAlg);
                }
                else {
                    NO_ARRAY_ERROR();
                }

            }
        });

        selectionSortButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (arr != null) {
                    //Keep track of how long it took to run the program and
                    // how many comparisons were done, how many movements.
                    long start_time, timetaken;

                    //Create temparr so the other array remains untouched.
                    int temparr[] = arr.clone();

                    start_time = System.nanoTime();
                    ReturnInformation data = SortAlg.SelectionSort(temparr);

                    //Caclulate time taken to complete selectionsort
                    timetaken = System.nanoTime() - start_time;

                    //Update the results
                    displayInformation(data, "Selection", timetaken,
                            temparr.length);

                    //Find new winning algorithm
                    WinningAlgorithm competingAlg = new WinningAlgorithm
                            ("Selection", data.getComparisons()
                                    + data.getMovements());

                    winAlg = determineWinningAlgorithm(winAlg, competingAlg);
                }
                else {
                    NO_ARRAY_ERROR();
                }
            }
        });

        mergeSortButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (arr != null) {
                    //Keep track of how long it took to run the program and
                    // how many comparisons were done, how many movements.
                    long start_time, timetaken;

                    //Create temparr so the other array remains untouched.
                    int temparr[] = arr.clone();

                    start_time = System.nanoTime();
                    ReturnInformation data = SortAlg.MergeSort(temparr);

                    //Calculate the time taken to complete MergeSort
                    timetaken = System.nanoTime() - start_time;



                    //Update the results
                    displayInformation(data, "Merge", timetaken,
                            temparr.length);

                    //Find new winning algorithm
                    WinningAlgorithm competingAlg = new WinningAlgorithm
                            ("Merge", data.getComparisons()
                                    + data.getMovements());

                    winAlg = determineWinningAlgorithm(winAlg, competingAlg);
                }
                else {
                    NO_ARRAY_ERROR();
                }
            }
        });

        quickSortButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (arr != null) {
                    //Keep track of how long it took to run the program and
                    // how many comparisons were done, how many movements.
                    long start_time, timetaken;

                    //Create temparr so the other array remains untouched.
                    int temparr[] = arr.clone();

                    start_time = System.nanoTime();
                    ReturnInformation data = SortAlg.quickSort(temparr);

                    //Calculate the time taken to complete quick sort
                    timetaken = System.nanoTime() - start_time;

                    //Update the results
                    displayInformation(data, "Quick", timetaken,
                            temparr.length);

                    //Find new winning algorithm
                    WinningAlgorithm competingAlg = new WinningAlgorithm
                            ("Quick", data.getComparisons()
                                    + data.getMovements());

                    winAlg = determineWinningAlgorithm(winAlg, competingAlg);
                }
                else {
                    NO_ARRAY_ERROR();
                }
            }
        });

        radixSortButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (arr != null) {
                    //Keep track of how long it took to run the program and
                    // how many comparisons were done, how many movements.
                    long start_time, timetaken;

                    //Create temparr so the other array remains untouched.
                    int temparr[] = arr.clone();

                    start_time = System.nanoTime();
                    ReturnInformation data = SortAlg.radixSort(temparr);

                    //Calculate the time taken to complete quick sort
                    timetaken = System.nanoTime() - start_time;

                    //Update the results
                    displayInformation(data, "Radix", timetaken, temparr.length);

                    //Find new winning algorithm
                    WinningAlgorithm competingAlg = new WinningAlgorithm
                            ("Radix", data.getComparisons()
                                    + data.getMovements());

                    winAlg = determineWinningAlgorithm(winAlg, competingAlg);
                }
                else {
                    NO_ARRAY_ERROR();
                }
            }
        });

        heapSortButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (arr != null) {
                    //Keep track of how long it took to run the program and
                    // how many comparisons were done, how many movements.
                    long start_time, timetaken;

                    //Create temparr so the other array remains untouched.
                    int temparr[] = arr.clone();

                    start_time = System.nanoTime();
                    ReturnInformation data = SortAlg.heapSort(temparr);

                    //Calculate the time taken to complete quick sort
                    timetaken = System.nanoTime() - start_time;

                    //Update the results
                    displayInformation(data, "Heap", timetaken, temparr.length);

                    //Find new winning algorithm
                    WinningAlgorithm competingAlg = new WinningAlgorithm
                            ("Heap", data.getComparisons()
                                    + data.getMovements());

                    winAlg = determineWinningAlgorithm(winAlg, competingAlg);
                }
                else {
                    NO_ARRAY_ERROR();
                }
            }
        });


    }
    public static void main(String args[]) {
        //Frame "App" will display the GUI to the user
        JFrame frame = new JFrame("Algorithm Analyser");
        frame.setContentPane(new AlgorithmAnalyser().panel1);
        //Simply "x" out of the program to leave
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }

    public static void NO_ARRAY_ERROR() {
        JOptionPane.showMessageDialog(new JFrame(),
                "Please specify an array size and how it should be ordered.");
    }




}