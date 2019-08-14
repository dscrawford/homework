abstract public class CatalogItem implements Comparable<CatalogItem>{
    private String title;
    private double price;

    CatalogItem () {}

    CatalogItem (String title, double price) {
        this.title = title;
        this.price = price;
    }

    abstract public String toString(); //Returns all the data in a format

    abstract public int getID(); //Returns either its ISBN or its dvdcode

    public String getTitle() {
        return this.title;
    }

    public double getPrice() {
        return this.price;
    }

    public int compareTo(CatalogItem compareItem) {
        double compare_val = compareItem.getPrice();
        return (int)Math.floor(this.price - compare_val);
    }
}
