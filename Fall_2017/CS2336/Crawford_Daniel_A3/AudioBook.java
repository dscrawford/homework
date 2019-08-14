public class AudioBook extends Book{
    private double runningTime; //Will be based on MINUTES
    public AudioBook(String title, double price, String author, int ISBN,
                     double runningTime) {
        super(title, price, author, ISBN);
        this.runningTime = runningTime;
    }
    public double getPrice() { //Audiobook gives a 10% discount
        return super.getPrice() - (super.getPrice() * 0.1);
    }
    public double getRunningTime() {
        return this.runningTime;
    }
    @Override
    public String toString() { //Appends runningtime to the end of the
        // toString in Book
        return super.toString() + "| RunningTime: " + this.runningTime;
    }
}
