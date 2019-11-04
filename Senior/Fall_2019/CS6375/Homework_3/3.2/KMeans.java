/*** Author :Vibhav Gogate
 The University of Texas at Dallas
 *****/
/*** Student: Daniel Crawford
 * Implemented the kmeans function as well as dist, and hasConverged
 */


import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;


public class KMeans {
    public static void main(String [] args){
        if (args.length < 3){
            System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
            return;
        }
        try{
            BufferedImage originalImage = ImageIO.read(new File(args[0]));
            int k=Integer.parseInt(args[1]);
            BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
            ImageIO.write(kmeansJpg, "jpg", new File(args[2]));
        }catch(IOException e){
            System.out.println(e.getMessage());
        }
    }

    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
        int w=originalImage.getWidth();
        int h=originalImage.getHeight();
        BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
        Graphics2D g = kmeansImage.createGraphics();
        g.drawImage(originalImage, 0, 0, w,h , null);
        // Read rgb values from the image
        int[] rgb=new int[w*h];
        int count=0;
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                rgb[count++]=kmeansImage.getRGB(i,j);
            }
        }
        // Call kmeans algorithm: update the rgb values
        kmeans(rgb,k);

        // Write the new rgb values to the image
        count=0;
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                kmeansImage.setRGB(i,j,rgb[count++]);
            }
        }
        return kmeansImage;
    }

    //Euclidian distance
    private static float dist(Color x, Color y) {
        float x1 = x.getRed();float x2 = x.getBlue();float x3 = x.getGreen();
        float y1 = y.getRed();float y2 = y.getBlue();float y3 = y.getGreen();
        return (float) Math.sqrt(Math.pow(x1 - y1, 2) + Math.pow(x2 - y2, 2) + Math.pow(x3 - y3, 2));
    }

    //Tests if an iteration did not provide any changes.
    private static boolean hasConverged(List<Color> X, List<Color> Y) {
        if (X.size() != Y.size()) return false;
        boolean allSame = true;
        for (int i = 0; i < X.size(); ++i) {
            allSame = X.get(i).getRGB() == Y.get(i).getRGB();
        }
        return allSame;
    }
    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    // kmeans(rgb, k):
    //  c <- k random data points from rgb
    //  clusterList <- k separate lists which hold Colors
    //  while hasnt converged:
    //   for each point in rgb
    //    select point in cluster that is closest to point
    //    add point to clusterList with closest cluster
    //   for each cluster in c:
    //    c[i] <- mean(clusterList[i])
    //  for each point in rgb:
    //    set point to closest cluster in c
    // Time complexity to find new mean: O(D * k * f) where D is data length,
    // k is clusters, and f is the number of features.
    // Total time complexity depends on how long it takes to converge.
    private static void kmeans(int[] rgb, int k) {
        ArrayList<Color> rgbList = new ArrayList<>();
        for (int value : rgb) rgbList.add(new Color(value));
        ArrayList<Color> copy = new ArrayList<>(rgbList);
        Collections.shuffle(copy);
        List<Color> c = copy.subList(0, k);

        ArrayList<ArrayList<Color>> clustersList = new ArrayList<>(c.size());
        for (int i = 0; i < k; ++i) clustersList.add(new ArrayList<>());

        List<Color> prev = new ArrayList<>();
        while (!hasConverged(prev, c)) {
            prev = new ArrayList<>(c);
            for (Color x : rgbList) {
                int closestI = 0;
                float closest = -1;
                for (int i = 0; i < c.size(); ++i) {
                    Color y = c.get(i);
                    float dist = dist(x, y);
                    if (dist < closest || closest == -1) {
                        closestI = i;
                        closest = dist;
                    }
                }
                clustersList.get(closestI).add(x);
            }

            for (int i = 0; i < c.size(); ++i) {
                float sumR = 0, sumG = 0, sumB = 0;
                int n = clustersList.get(i).size();
                for (Color x : clustersList.get(i)) {
                    sumR += x.getRed();
                    sumB += x.getBlue();
                    sumG += x.getGreen();
                }
                float meanR = sumR / n;
                float meanG = sumG / n;
                float meanB = sumB / n;
                c.set(i, new Color((int) meanR, (int) meanG, (int) meanB));
            }
        }

        for (int j = 0; j < rgb.length; ++j) {
            Color x = new Color(rgb[j]);
            int closestI = 0;
            float closest = -1;
            for (int i = 0; i < c.size(); ++i) {
                Color y = c.get(i);
                float dist = dist(x, y);
                if (dist < closest || closest == -1) {
                    closestI = i;
                    closest = dist;
                }
            }
            rgb[j] = c.get(closestI).getRGB();
        }
    }
}
