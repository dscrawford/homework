//Made by Daniel Crawford on 3/3/2018(dsc160130)
public class Book {
    private String name;
    private String author;
    Book(String name, String author) {
        setAuthor(author);
        setName(name);
    }
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public void display() {
        System.out.println("Name: " + name + ", Author: " + author);
    }
}
