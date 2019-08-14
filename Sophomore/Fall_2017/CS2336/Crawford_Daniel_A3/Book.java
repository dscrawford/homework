public class Book extends CatalogItem {
    private String author;
    private int ISBN;
    Book() {} //This constructor only exists for inheritance purposes.
    public Book(String title, double price, String author, int ISBN) {
        super(title,price);
        this.author= author;
        this.ISBN = ISBN;
    }
    public String toString() { //Returns the title, author, price and ISBN #
        return "Title: " + getTitle() + " | Author: " + getAuthor() + " | " +
                "Price: " + getPrice() + " | ISBN: " + getID();
    }
    public String getAuthor() {
        return this.author;
    }
    public int getID() {
        return ISBN;
    } //get ISBN
}