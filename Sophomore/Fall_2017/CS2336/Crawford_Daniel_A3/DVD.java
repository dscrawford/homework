public class DVD extends CatalogItem {
    private String director;
    private int year;
    private int dvdcode;
    public DVD(String title, double price, String director, int year, int
            dvdcode) { //Assigns all variables in parameters
        super(title,price);
        this.director = director;
        this.year = year;
        this.dvdcode = dvdcode;
    }
    public String toString() { //Returns the title, author, price and dvd code
        return "Title: " + super.getTitle() + " | Director: " + getDirector()
                + " |" + " Price: " + super.getPrice() + " | Year: " +
                getYear() + " | DvdCode: " + getID();
    }
    public int getID() {
        return this.dvdcode;
    }
    public int getYear() {
        return this.year;
    }
    public String getDirector() {
        return this.director;
    }
}
