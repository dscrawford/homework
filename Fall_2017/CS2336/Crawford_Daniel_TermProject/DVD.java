public class DVD extends CatalogItem {
    private String director;
    private int year;
    private int dvdcode;

    public DVD(String title, double price, String director, int year, int
            dvdcode) { //Assigns all variables in parameters
        super(title,price * 0.8);
        this.director = director;
        this.year = year;
        this.dvdcode = dvdcode;
    }

    public String toString() { //Returns the title, author, price and dvd code
        String result = String.format("Title: %25s | Director: %18s | Price: " +
                        "%.2f | Year: %6d | DvdCode: %6d", getTitle(),
                getDirector(), getPrice(), getYear(), getID());
        return result;
    }

    public int getID() {
        return this.dvdcode;
    }

    public int getYear() {
        return this.year;
    }

    public String getDirector() { return this.director; }
}
