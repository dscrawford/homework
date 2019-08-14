//Made by Daniel Crawford on 02/14/2019 (dsc160130@utdallas.edu)
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.math.*;

public class Exercise_2_1 extends Frame {
    public static void main(String args[]) {
        new Exercise_2_1();
    }

    Exercise_2_1() {
        super("Isotropic mapping mode");
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent windowEvent) {
                super.windowClosing(windowEvent);
                System.exit(0);
            }
        });
        add("Center", new mouseSquare());
        setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
        setVisible(true);
    }
}

class mouseSquare extends Canvas {
    private int centerX, centerY, mousePress = 0;
    float pixelSize, rWidth = 10.0F, rHeight = 10.0F,
            xP1, yP1, xP2,  yP2;
    boolean ready = false, messageDrawn = false;

    mouseSquare() {
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent mouseEvent) {
                mouseIsPressed();
                switch (mousePress) {
                    case 0:
                            ready = false;
                            repaint();
                            break;
                    case 1: xP1 = fx(mouseEvent.getX());
                            yP1  = fy(mouseEvent.getY());
                            repaint();
                            break;
                    case 2: xP2 = fx(mouseEvent.getX());
                            yP2 = fy(mouseEvent.getY());
                            ready = true;
                            repaint();
                            break;
                    default: System.out.println("what");
                }

            }
        });
    }

    void initgr() {
        Dimension d = getSize();
        int maxX = d.width - 1, maxY = d.height - 1;
        pixelSize = Math.max(rWidth / maxX, rHeight / maxY);
        centerX = maxX / 2; centerY = maxY / 2;
    }

    void mouseIsPressed() {
        mousePress = (mousePress + 1) % 3;
    }

    int iX(float x) {return Math.round( centerX + x / pixelSize);}
    int iY(float y) {return Math.round( centerY - y / pixelSize);}
    float fx(int x) {return (x - centerX) * pixelSize;}
    float fy(int y) {return (centerY - y) * pixelSize;}
    float dist(float x1, float y1, float x2, float y2) {
            return (float) Math.sqrt( Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }


    public void paint(Graphics g) {
        initgr();

        if (ready) {
            float squareLen = dist(xP1, yP1, xP2, yP2);

            int aX = iX(xP1), aY = iY(yP1), bX = iX(xP2), bY = iY(yP2);
            int left = iX(-rWidth / 2), right = iX(rWidth / 2),
                    bottom = iY(-rHeight / 2), top = iY(rHeight / 2),
                    xMiddle = iX(0), yMiddle = iY(0);

            int xdist = (bX - aX);
            int ydist = (bY - aY);
            int dX = aX + ydist, dY = aY - xdist, cX = bX + ydist, cY = bY - xdist;

            g.drawLine(aX,aY,bX,bY);
            g.drawLine(bX,bY,cX,cY);
            g.drawLine(cX,cY,dX,dY);
            g.drawLine(dX,dY,aX,aY);
        }
        else if(!ready && !messageDrawn) {
            g.drawString("cleared", iX(0), iX(0));
            messageDrawn = true;
        }
        else
            messageDrawn = false;
    }
}