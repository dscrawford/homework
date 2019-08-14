public class Book extends CatalogItem {
    private String author;
    private int ISBN;

    Book() {} //This constructor only exists for inheritance purposes.

    public Book(String title, double price, String author, int ISBN) {
        super(title,price * 0.9);
        this.author= author;
        this.ISBN = ISBN;
    }

    public String toString() { //Returns the title, author, price and ISBN #
        String result = String.format("Title: %25s | Author: %20s | Price: " +
                "%.2f | ISBN: %5d", getTitle(), getAuthor(), getPrice(),
                getID());
        return result;
    }

    public String getAuthor() {
        return this.author;
    }

    public int getID() {
        return ISBN;
    } //get ISBN
}