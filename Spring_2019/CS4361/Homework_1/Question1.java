//Made by Daniel Crawford on 1/27/2019
import java.awt.*;
import java.awt.event.*;

public class Question1 extends Frame {
    public static void main(String args[]) {
        new Question1();
    }

    Question1() {
        super("Question1");
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                System.exit(0);
            }
        });
        setSize(300,150);
        add("Center", new CvRedRect());
        setVisible(true);
    }
}

class CvRedRect extends Canvas {
    public void paint (Graphics g) {
        Dimension d = getSize();
        int centerX = (d.width - 1)/2, centerY = (d.height - 1)/2;

        //For points of the square
        int aX = centerX/2, aY = centerY/2,
                bX = centerX * 3/2, bY = centerY/2,
                cX = centerX * 3/2, cY = centerY * 3/2,
                dX = centerX/2, dY = centerY * 3/2;

        for (int i = 0; i < 20; ++i) {
            g.drawLine(aX,aY,bX,bY);
            g.drawLine(bX,bY,cX,cY);
            g.drawLine(cX,cY,dX,dY);
            g.drawLine(dX,dY,aX,aY);

            int tempX = aX, tempY = aY;
            aX = (aX + bX) / 2;
            aY = (aY + bY) / 2;
            bX = (bX + cX) / 2;
            bY = (bY + cY) / 2;
            cX = (cX + dX) / 2;
            cY = (cY + dY) / 2;
            dX = (dX + tempX) / 2;
            dY = (dY + tempY) / 2;
        }

    }
}