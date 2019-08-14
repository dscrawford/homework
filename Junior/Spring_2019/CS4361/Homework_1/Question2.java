//Made by Daniel Crawford on 1/27/2019 (dsc160130@utdallas.edu
import java.awt.*;
import java.awt.event.*;
import java.math.*;

public class Question2 extends Frame{
    public static void main(String args[]) {
        new Question2();
    }

    Question2() {
        super("Isotropic mapping mode");
        addWindowListener(new WindowAdapter() {
                              @Override
                              public void windowClosing(WindowEvent e) {
                                  super.windowClosing(e);
                                  System.exit(0);
                              }
                          });
        setSize(400,300);
        add("Center", new CvTetris());
        setVisible(true);
    }
}

class CvTetris extends Canvas {
    private int startMainRectX, startMainRectY, mainRectX, mainRectY,
                startQuitBoxX, startQuitBoxY, quitBoxX, quitBoxY;
    boolean inMainBox, pauseBoxPainted;

    CvTetris() {
        pauseBoxPainted = false;
        inMainBox = false;
        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();
                if (x >= startQuitBoxX && x <= startQuitBoxY + quitBoxX && y >= startQuitBoxY
                        && y <= startQuitBoxY + quitBoxY)
                    System.exit(0);
            }
        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();
                inMainBox = (x >= startMainRectX && x <= startMainRectX + mainRectX && y >= startMainRectY
                        && y <= startMainRectY + mainRectY);
                if (!pauseBoxPainted && inMainBox)
                    repaint();
                if (!inMainBox && pauseBoxPainted) {
                    repaint();
                }
            }
        });
    }

    public void paint(Graphics g) {
        Dimension d = getSize();
        int maxX = d.width - 1, maxY = d.height - 1, minDist = Math.min(maxX, maxY);
        int squareLen = minDist / 20;

        mainRectX = squareLen * 10;
        mainRectY = squareLen * 20;
        startMainRectX = maxX * 2 / 16;
        startMainRectY = maxY / 16;

        int startNextRectX = startMainRectX + mainRectX + squareLen, startNextRectY = startMainRectY;

        quitBoxX = squareLen * 3;
        quitBoxY = squareLen * 2;
        startQuitBoxX = startNextRectX;
        startQuitBoxY = startNextRectY + 18 * squareLen;

        //Draw main box
        g.drawRect(startMainRectX, startMainRectY, mainRectX, mainRectY);
        g.drawRect(startNextRectX, startNextRectY, squareLen * 5, squareLen * 4);
        g.drawRect(startQuitBoxX, startQuitBoxY, quitBoxX, quitBoxY);

        CvTetrisPieces.Square(g, startMainRectX, startMainRectY, 4, 2, squareLen);
        CvTetrisPieces.Squiggly(g, startMainRectX, startMainRectY, 6, 19, squareLen);
        CvTetrisPieces.LBlock(g, startMainRectX, startMainRectY, 8, 19, squareLen);
        CvTetrisPieces.ReverseLBlock(g, startNextRectX, startNextRectY, 1, 2, squareLen);

        g.setFont(new Font("TimesRoman", Font.PLAIN, squareLen));
        g.drawString("Level:   1", startNextRectX, startNextRectY + 7 * squareLen);
        g.drawString("Lines:   0", startNextRectX, startNextRectY + 8 * squareLen);
        g.drawString("Score:   0", startNextRectX, startNextRectY + 9 * squareLen);
        g.setFont(new Font("TimesRoman", Font.BOLD, squareLen));
        g.drawString("QUIT", startNextRectX,
                startNextRectY + 19 * squareLen);

        if (inMainBox) { //draw paint box
            Color c = g.getColor();
            g.setColor(Color.blue);
            g.drawRect(startMainRectX + 3 * squareLen, startMainRectY + 10 * squareLen,
                    squareLen * 4, squareLen * 2);
            g.setFont(new Font("TimesRoman", Font.BOLD, squareLen));
            g.drawString("PAUSE", startMainRectX + 3 * squareLen, startMainRectY + 11* squareLen);
            pauseBoxPainted = true;
        }
        else
            pauseBoxPainted = false;
    }
}

class CvTetrisPieces extends Canvas {
    public static void Square(Graphics g, int startX, int startY, int x, int y, int squareLen) {
        Color c = g.getColor();
        g.setColor(Color.green);
        g.fillRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.green);
        g.fillRect(startX + (x + 1) * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 1) * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.green);
        g.fillRect(startX + (x + 1) * squareLen, startY + (y + 1) * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 1) * squareLen, startY + (y + 1) * squareLen, squareLen, squareLen);

        g.setColor(Color.green);
        g.fillRect(startX + x * squareLen, startY + (y + 1) * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + x * squareLen, startY + (y + 1) * squareLen, squareLen, squareLen);

        g.setColor(c);
    }

    public static void Squiggly(Graphics g, int startX, int startY, int x, int y, int squareLen) {
        Color c = g.getColor();
        g.setColor(Color.yellow);
        g.fillRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.yellow);
        g.fillRect(startX + (x + 1) * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 1) * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.yellow);
        g.fillRect(startX + (x + 1) * squareLen, startY + (y - 1) * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 1) * squareLen, startY + (y - 1) * squareLen, squareLen, squareLen);

        g.setColor(Color.yellow);
        g.fillRect(startX + (x + 2) * squareLen, startY + (y - 1) * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 2) * squareLen, startY + (y - 1) * squareLen, squareLen, squareLen);

        g.setColor(c);
    }

    public static void LBlock(Graphics g, int startX, int startY, int x, int y, int squareLen) {
        Color c = g.getColor();
        g.setColor(Color.blue);
        g.fillRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.blue);
        g.fillRect(startX + (x + 1) * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 1) * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.blue);
        g.fillRect(startX + (x + 1) * squareLen, startY + (y - 1) * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 1) * squareLen, startY + (y - 1) * squareLen, squareLen, squareLen);

        g.setColor(Color.blue);
        g.fillRect(startX + (x + 1) * squareLen, startY + (y - 2) * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 1) * squareLen, startY + (y - 2) * squareLen, squareLen, squareLen);

        g.setColor(c);
    }

    public static void ReverseLBlock(Graphics g, int startX, int startY, int x, int y, int squareLen) {
        Color c = g.getColor();
        g.setColor(Color.red);
        g.fillRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.red);
        g.fillRect(startX + (x + 1) * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 1) * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.red);
        g.fillRect(startX + (x + 2) * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 2) * squareLen, startY + y * squareLen, squareLen, squareLen);

        g.setColor(Color.red);
        g.fillRect(startX + (x + 2) * squareLen, startY + (y - 1) * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + (x + 2) * squareLen, startY + (y - 1) * squareLen, squareLen, squareLen);

        g.setColor(c);
    }
}