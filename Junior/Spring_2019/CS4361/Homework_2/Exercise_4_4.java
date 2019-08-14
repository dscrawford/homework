//Made by Daniel Crawford on 02/14/2019 (dsc160130@utdallas.edu)
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.*;
import java.util.ArrayList;

public class Exercise_4_4 extends Frame {
    public static void main(String args[]) {
        new Exercise_4_4();
    }

    Exercise_4_4() {
        super("Grid Homework");
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent windowEvent) {
                super.windowClosing(windowEvent);
                System.exit(0);
            }
        });
        ReadGridFile read = new ReadGridFile();
        ArrayList<ArrayList<Integer>> queries = read.getResults();

        add("Center", new Grid(queries));
        setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
        setVisible(true);
    }


}

//Class simply opens up "input.txt" and returns its results in a two dimensional arraylist
class ReadGridFile {
    private ArrayList<ArrayList<Integer>> results;

    ReadGridFile() {
        results = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader("input.txt"));
            String line;
            boolean firstIter = true;
            int LinesLeft = 0;

            //Read line by line from input.txt
            while ( (line = br.readLine()) != null) {
                //arr holds the split array, where the space character is the delimiter
                String[] arr;
                arr = line.split(" ");

                //If it is the first iteration, then it should indicate the amount of lines drawn
                if (firstIter) {
                    if (arr.length > 1) {
                        System.out.println("ERROR: Incorrect File format(First line should contain one integer");
                        System.exit(1);
                    }
                    LinesLeft = Integer.parseInt(arr[0]);
                    firstIter = false;
                }
                //Otherwise, it is cooridinates to be drawn
                else {
                    ArrayList<Integer> temp = new ArrayList<>();
                    //If it is a line
                    if (LinesLeft > 0) {
                        for (int i = 0; i <= 3; ++i)
                            temp.add(Integer.parseInt(arr[i]));
                        results.add(temp);
                        LinesLeft--;
                    }
                    //If it is a circle
                    else {
                        for (int i = 0; i <= 2; ++i)
                            temp.add(Integer.parseInt(arr[i]));
                        results.add(temp);
                    }
                }
            }
        }
        catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
    }

    public ArrayList<ArrayList<Integer>> getResults() {
        return results;
    }
}

class Grid extends Canvas {
    private int centerX, centerY, maxX, maxY, dGrid;
    float pixelSize, rWidth = 10.0F, rHeight = 10.0F;
    boolean ready = false, messageDrawn = false;
    ArrayList<ArrayList<Integer>> queries;

    Grid(ArrayList<ArrayList<Integer>> queries) {
        this.queries = queries;
    }

    void initgr() {
        Dimension d = getSize();
        maxX = d.width - 1;
        maxY = d.height - 1;
        dGrid = 10;
        pixelSize = Math.max(rWidth / maxX, rHeight / maxY);
        centerX = maxX / 2; centerY = maxY / 2;
    }

    int iX(float x) {return Math.round( centerX + x / pixelSize);}
    int iY(float y) {return Math.round( centerY - y / pixelSize);}


    public void paint(Graphics g) {
        initgr();
        drawGrid(g);

        for (ArrayList<Integer> list : queries) {
            if (list.size() == 3)
                drawCircle(g,list.get(0),list.get(1),list.get(2));
            else
                drawLine(g, list.get(0),list.get(1), list.get(2), list.get(3));
        }
    }

    //Bresenhem Line algorithm
    private void drawLine(Graphics g, int xP, int yP, int xQ, int yQ) {
        int x = xP, y = yP, d = 0, dx = xQ - xP, dy = yQ - yP, c, m,
                xInc = 1, yInc = 1;
        if (dx < 0) {xInc = -1; dx = -dx;}
        if (dy < 0) {yInc = -1; dy = -dy;}
        if (dy <= dx) {
            c = 2 * dx; m = 2 * dy;
            if (xInc < 0) dx++;
            for(;;) {
                putPixel(g,x,y);
                if (x == xQ) break;
                x += xInc;
                d += m;
                if (d >= dx) { y += yInc; d -= c; }
            }
        }
        else {
            c = 2 * dy; m = 2 * dx;
            if (yInc < 0) dy++;
            for(;;) {
                putPixel(g,x,y);
                if (y == yQ) break;
                y += yInc;
                d += m;
                if (d >= dy) { x += xInc; d -= c; }
            }
        }
    }

    //Bresenhem Circle algorithm
    private void drawCircle(Graphics g, int xC, int yC, int r) {
        int x = 0, y = r, u = 1, v = 2 * r - 1, e = 0;
        while (x < y) {
            putPixel(g, xC + x, yC + y);
            putPixel(g, xC + y, yC - x);
            putPixel(g, xC - x, yC - y);
            putPixel(g, xC - y, yC + x);
            x++; e += u; u += 2;
            if (v < 2 * e) {y--; e -= v; v-= 2;}

            if (x > y) break;
            putPixel(g, xC + y, yC + x);
            putPixel(g, xC + x, yC - y);
            putPixel(g, xC - y, yC - x);
            putPixel(g, xC - x, yC + y);
        }
    }

    private void putPixel(Graphics g, int x, int y) {
        int xGrid = x * dGrid, yGrid = y * dGrid, radius = dGrid / 2;
        g.drawOval(xGrid - radius, yGrid - radius, dGrid, dGrid);
    }

    private void drawGrid(Graphics g) {
        for (int x = dGrid; x < maxX; x += dGrid) {
            for (int y = dGrid; y < maxY; y += dGrid) {
                g.drawLine(x,y, x, y);
            }
        }
    }
}