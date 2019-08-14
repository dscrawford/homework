import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.geom.Point2D;

public class messaround extends Frame {
    public static void main(String args[]) {
        new messaround();
    }
    messaround() {
        super("Animation (double buffering)");
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        add("Center", new CvAnim());
        Dimension dim = getToolkit().getScreenSize();
        setSize(dim.width / 2, dim.height / 2);
        setLocation(dim.width / 4, dim.height / 4);
        setVisible(true);
    }
}

class CvAnim extends Canvas implements Runnable {
    float rWidth = 10.0F, rHeight = 10.0F, xC, yC, pixelSize;
    int centerX, centerY, w, h;
    Dimension d;
    Image image;
    Graphics gImage;

    float alpha = 0;
    Thread thr = new Thread(this);

    public void run() {
        try {
            for(;;) {
                alpha += 0.01;
                repaint();
                Thread.sleep(1);
            }
        }
        catch (InterruptedException e) {

        }
    }

    CvAnim() {
        thr.start();
    }

    void initgr() {
        d = getSize();
        int maxX = d.width - 1, maxY = d.height - 1;
        pixelSize = Math.max(rWidth / maxX, rHeight / maxY);
        centerX = maxX / 2; centerY = maxY / 2;
        xC = rWidth / 2; yC = rHeight / 2;
    }

    int iX(float x) {return Math.round(centerX + x / pixelSize);}
    int iY(float y) {return Math.round(centerY + y / pixelSize);}
    public void update(Graphics g) {paint(g);}

    float rotate = 0.01F;
    public void paint (Graphics g) {
        initgr();
        if (w != d.width || h != d.height) {
            w = d.width; h = d.height;
            image = createImage(w, h);
            gImage = image.getGraphics();
        }

        d.getSize();


        gImage.clearRect(0,0,w,h);
        float r = rotate * Math.min(xC, yC),
                x = r * (float) Math.cos(alpha),
                y = r * (float) Math.sin(alpha);
        Point2D a1 = new Point(1,2);



        g.drawImage(image, 0, 0, null);
    }

    private Point2D RotateAroundPoint(Point2D center, Point2D rotater, int degree) {
        return new Point(1,2);
    }
}