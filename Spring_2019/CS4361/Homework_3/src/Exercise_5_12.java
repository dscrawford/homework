import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Arrays;

public class Exercise_5_12 extends Frame {
    public static void main(String args[]) {
        new Exercise_5_12();
    }
    Exercise_5_12() {
        super("Animation (double buffering)");
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        add("Center", new CvTetris());
        Dimension dim = getToolkit().getScreenSize();
        setSize(dim.width / 2, dim.height / 2);
        setLocation(dim.width / 4, dim.height / 4);
        setVisible(true);
    }
}

/*
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
                Thread.sleep(5);
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

    public void paint (Graphics g) {
        initgr();
        if (w != d.width || h != d.height) {
            w = d.width; h = d.height;
            image = createImage(w, h);
            gImage = image.getGraphics();
        }
        float r = 0.8F * Math.min(xC, yC),
                x = r * (float) Math.cos(alpha),
                y = r * (float) Math.sin(alpha);

        gImage.clearRect(0,0,w,h);

        gImage.drawLine(iX(0), iY(0), iX(x), iY(y));
        g.drawImage(image, 0, 0, null);
    }
}
*/

class CvTetris extends Canvas implements Runnable {
    private float startMainRectX, startMainRectY, mainRectX, mainRectY,
            startQuitBoxX, startQuitBoxY, quitBoxX, quitBoxY,
            startNextRectX, startNextRectY,nextRectX, nextRectY, squareLen,
            rWidth = 40.0F, rHeight = 40.0F, xC, yC, pixelSize;

    private int centerX, centerY;
    private Dimension d;
    private Image image;
    int rot = 0;



    private Thread thr = new Thread(this);

    boolean inMainBox, pauseBoxPainted;

    boolean occupiedSpaces[][];
    boolean gameBoardChanged = false;

    CvTetris() {
        thr.start();
        occupiedSpaces = new boolean[10][20];
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 20; ++j)
                occupiedSpaces[i][j] = false;
        }
        oldpoints = new ArrayList<>();
        pauseBoxPainted = false;
        inMainBox = false;
        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int x = e.getX();
                int y = e.getY();
                if (x >= iX(startQuitBoxX) && x <= iX(startQuitBoxX + quitBoxX) && y >= iY(startQuitBoxY)
                        && y <= iY(startQuitBoxY + quitBoxY))
                    System.exit(0);

                boolean leftButton  = (e.getModifiersEx() & InputEvent.BUTTON1_DOWN_MASK) != 0;
                boolean rightButton = (e.getModifiersEx() & InputEvent.BUTTON1_DOWN_MASK) != 0;
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

        addMouseWheelListener(new MouseWheelListener() {
            @Override
            public void mouseWheelMoved(MouseWheelEvent mouseWheelEvent) {
                rot = (rot + mouseWheelEvent.getWheelRotation()) % 4;
                System.out.println(mouseWheelEvent.getWheelRotation());
                gameBoardChanged = true;
            }
        });
    }


    public void run() {
        try {
            for(;;) {
                repaint();
                Thread.sleep(25);
            }
        }
        catch (InterruptedException e) {
            System.out.println(e.getMessage());
        }
    }

    void initgr() {
        d = getSize();
        int maxX = d.width - 1, maxY = d.height - 1;
        pixelSize = Math.max(rWidth / maxX, rHeight / maxY);
        centerX = maxX / 2; centerY = maxY / 2;
        xC = rWidth / 2; yC = rHeight / 2;

        squareLen = 1.5F;

        startMainRectX = -5F;
        startMainRectY = -19F;
        mainRectX = squareLen * 10;
        mainRectY = squareLen * 20;


        startNextRectX = startMainRectX + mainRectX + squareLen;
        startNextRectY = startMainRectY;
        nextRectX = squareLen * 3;
        nextRectY = squareLen * 3;


        startQuitBoxX = startNextRectX;
        startQuitBoxY = startNextRectY + squareLen * 18;
        quitBoxX = squareLen * 3;
        quitBoxY = squareLen * 2;
    }

    int iX(float x) {return Math.round(centerX + x / pixelSize);}
    int iY(float y) {return Math.round(centerY + y / pixelSize);}
    float fx(int x) {return (x - centerX) * pixelSize;}
    float fy(int y) {return (centerY - y) * pixelSize;}
    int toDevice(float i) {return Math.round(i / pixelSize);}

    public void update(Graphics g) { paint(g);}

    ArrayList<Point> oldpoints;
    public void paint (Graphics g) {
        /*
        if (!oldpoints.isEmpty()) {
            for (int i = 0; i < oldpoints.size(); ++i)
                occupiedSpaces[oldpoints.get(i).x][oldpoints.get(i).y] = false;
            oldpoints = new ArrayList<>();
        }
        */

        if (gameBoardChanged) {
            gameBoardChanged = false;
            g.clearRect(0,0,d.width - 1, d.height - 1);
        }
        initgr();
        CvTetrisPieces tetris = new CvTetrisPieces(iX(startMainRectX), iY(startMainRectY), toDevice(squareLen), g);

        ArrayList<Point> points = tetris.makeLine(new Point(1, 0),rot, occupiedSpaces);
        if (points == null) {
            tetris.makeLBlock(new Point(1, 10), (rot - 1) % 4, occupiedSpaces);
            oldpoints = points;
        }
        //Draw main box
        g.clearRect(iX(startMainRectX),0,d.width - 1,iY(startMainRectY));
        createBoxes(g);
        createText(g);
        /*
        if (points == null) {
            tetris.makeLBlock(new Point(1, 10), (rot - 1) % 4, occupiedSpaces);
            oldpoints = points;
        }
        /*
        for (int i = 0; i < points.size(); ++i) {
            occupiedSpaces[]
        }
        */

        /*
        float r = 0.8F * Math.min(xC, yC),
                x = r * (float) Math.cos(alpha),
                y = r * (float) Math.sin(alpha);

        gImage.clearRect(0,0,w,h);

        gImage.drawLine(iX(0), iY(0), iX(x), iY(y));
        g.drawImage(image, 0, 0, null);
        */
    }

    void generateTetrisPiece(int x, int y, int tetrisPiece) {
        switch (tetrisPiece) {
            case 0:
                break;
            case 1: break;
            case 2: break;
            case 3: break;
            case 4: break;
            case 5: break;
            default: break;
        }
    }

    private void createBoxes(Graphics g) {
        g.drawRect(iX(startMainRectX), iY(startMainRectY), toDevice(mainRectX), toDevice(mainRectY));
        g.drawRect(iX(startNextRectX), iY(startNextRectY), toDevice(nextRectX), toDevice(nextRectY));
        g.drawRect(iX(startQuitBoxX), iY(startQuitBoxY), toDevice(quitBoxX), toDevice(quitBoxY));
    }

    private void createText(Graphics g) {
        g.setFont(new Font("TimesRoman", Font.PLAIN, toDevice(squareLen / 2)));
        g.drawString("Level:   1", iX(startNextRectX), iY(startNextRectY + 7 * squareLen));
        g.drawString("Lines:   0", iX(startNextRectX), iY(startNextRectY + 8 * squareLen));
        g.drawString("Score:   0", iX(startNextRectX), iY(startNextRectY + 9 * squareLen));

        g.setFont(new Font("TimesRoman", Font.BOLD, toDevice(squareLen)));
        g.drawString("QUIT", iX(startNextRectX), iY(startNextRectY + 19.5F * squareLen));

        if (inMainBox) { //draw paint box
            Color c = g.getColor();
            g.setColor(Color.blue);
            g.drawRect(iX(startMainRectX + 3 * squareLen), iY(startMainRectY + 10 * squareLen),
                    toDevice(squareLen * 4), toDevice(squareLen * 2));
            g.setFont(new Font("TimesRoman", Font.BOLD, toDevice(squareLen)));
            g.drawString("PAUSE", iX(startMainRectX + 3 * squareLen), iY(startMainRectY + 11* squareLen));
            pauseBoxPainted = true;
        }
        else
            pauseBoxPainted = false;
    }

    private boolean checkCollision(int x, int y) {
        return occupiedSpaces[x][y];
    }

    private void occupySpace(int x, int y) {
        occupiedSpaces[x][y] = true;
    }

    private void deoccupySpace(int x, int y) {
        occupiedSpaces[x][y] = false;
    }
}

class CvTetrisPieces extends Canvas {
    private int startX, startY, squareLen;
    Graphics2D g;

    CvTetrisPieces(int startX, int startY, int squareLen, Graphics g) {
        this.startX = startX; this.startY = startY;
        this.g = (Graphics2D) g.create();
        this.squareLen = squareLen;
    }

    private Point rotate(Point center, Point rotater, float angle) {
        float cx = center.x, cy = center.y,
                x = rotater.x, y = rotater.y;
        Point point = new Point(
                (int) Math.round(cx + (x - cx)*Math.cos(Math.toRadians(angle)) - (y - cy) * Math.sin(Math.toRadians(angle))),
                (int) Math.round(cy + (x - cx)*Math.sin(Math.toRadians(angle)) + (y - cy) * Math.cos(Math.toRadians(angle))));
        return point;
    }

    private boolean inGameBox(Point a, boolean collisions[][]) {
        int x = collisions.length;
        int y = collisions[0].length;
        System.out.println("a.x: " + a.x + ", a.y: " + a.y);
        System.out.println((a.x < x && a.x >= 0 && a.y >= 0 && a.y < y));
        return a.x < x && a.x >= 0 && a.y < y;
    }

    private boolean hasNoCollision(Point a, Point b, Point c, Point d, boolean collisions[][]) {
        if (inGameBox(a,collisions) && inGameBox(b,collisions) && inGameBox(c,collisions) && inGameBox(d,collisions))
            return !checkCollision(a,collisions) && !checkCollision(b,collisions)
                    && !checkCollision(c,collisions) && !checkCollision(d,collisions);
        return false;
    }

    private boolean checkCollision(Point a, boolean[][] collisions) {
        if (a.y < 0)
            return true;
        else
            return collisions[a.x][a.y];
    }
    private boolean generateTetrimino(ArrayList<Point> points, Color c, boolean[][] collisions) {
        if (hasNoCollision(points.get(0),points.get(1),points.get(2),points.get(3),collisions)) {
            drawSquare(points.get(0), c);
            drawSquare(points.get(1), c);
            drawSquare(points.get(2), c);
            drawSquare(points.get(3), c);
            return true;
        }
        return false;
    }

    private void drawSquare(Point coor, Color c) {
        int x = coor.x, y = coor.y;
        g.setColor(c);
        g.fillRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);
        g.setColor(Color.black);
        g.drawRect(startX + x * squareLen, startY + y * squareLen, squareLen, squareLen);
    }

    public ArrayList<Point> makeSquare(Point coordinates, int rot, boolean collisions[][]) {
        ArrayList<Point> points = new ArrayList<>();

        points.add(coordinates);
        points.add(new Point(coordinates.x + 1, coordinates.y));
        points.add(new Point(coordinates.x, coordinates.y + 1));
        points.add(new Point(coordinates.x + 1, coordinates.y));

        if (generateTetrimino(points,Color.green, collisions))
            return points;

        return null;
    }

    public ArrayList<Point> makeSquiggly(Point cooridinates, int rot, boolean collisions[][]) {
        rot = -1 * Math.abs(rot % 2);

        ArrayList<Point> points = new ArrayList<>();

        points.add(cooridinates);
        points.add(rotate(points.get(0),new Point(cooridinates.x - 1, cooridinates.y), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x - 1, cooridinates.y - 1), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x - 2, cooridinates.y - 1), rot * 90));

        if (generateTetrimino(points,Color.yellow, collisions))
            return points;

        return null;
    }

    public ArrayList<Point> makeRSquiggly(Point cooridinates, int rot, boolean[][] collisions) {
        Color c = new Color(59, 182,230);
        rot = -1 * Math.abs(rot % 2);

        ArrayList<Point> points = new ArrayList<>();

        points.add(cooridinates);
        points.add(rotate(points.get(0),new Point(cooridinates.x + 1, cooridinates.y), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x + 1, cooridinates.y - 1), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x + 2, cooridinates.y - 1), rot * 90));

        if (generateTetrimino(points,c, collisions))
            return points;

        return null;
    }
    public ArrayList<Point> makeLBlock(Point cooridinates, int rot, boolean[][] collisions) {
        Color c = new Color(230, 35, 33);

        ArrayList<Point> points = new ArrayList<>();

        points.add(cooridinates);
        points.add(rotate(points.get(0),new Point(cooridinates.x, cooridinates.y - 1), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x, cooridinates.y - 2), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x + 1, cooridinates.y - 2), rot * 90));

        if (generateTetrimino(points,c, collisions))
            return points;

        return null;
    }

    public ArrayList<Point> makeRLBlock(Point cooridinates, int rot, boolean[][] collisions) {
        Color c = new Color(221, 67, 230);

        ArrayList<Point> points = new ArrayList<>();

        points.add(cooridinates);
        points.add(rotate(points.get(0),new Point(cooridinates.x, cooridinates.y - 1), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x, cooridinates.y - 2), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x - 1, cooridinates.y - 2), rot * 90));

        if (generateTetrimino(points,c, collisions))
            return points;

        return null;
    }

    public ArrayList<Point> makeTBlock(Point cooridinates, int rot, boolean[][] collisions) {
        Color c = new Color(230, 137, 10);

        ArrayList<Point> points = new ArrayList<>();

        points.add(cooridinates);
        points.add(rotate(points.get(0),new Point(cooridinates.x - 1, cooridinates.y), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x + 1, cooridinates.y), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x, cooridinates.y - 1), rot * 90));

        if (generateTetrimino(points,c, collisions))
            return points;

        return null;
    }

    public ArrayList<Point> makeLine(Point cooridinates, int rot, boolean[][] collisions) {
        Color c = new Color(42, 0, 230);

        ArrayList<Point> points = new ArrayList<>();

        points.add(cooridinates);
        points.add(rotate(points.get(0),new Point(cooridinates.x, cooridinates.y - 1), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x, cooridinates.y - 2), rot * 90));
        points.add(rotate(points.get(0),new Point(cooridinates.x, cooridinates.y - 3), rot * 90));

        if (generateTetrimino(points,c, collisions))
            return points;

        return null;
    }
}
