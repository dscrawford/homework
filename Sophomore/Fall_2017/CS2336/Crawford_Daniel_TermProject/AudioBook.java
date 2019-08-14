public class AudioBook extends Book{
    private double runningTime; //Will be based on MINUTES

    public AudioBook(String title, double price, String author, int ISBN,
                     double runningTime) {
        //Divided by 0.9 to cut off discount in Book.
        super(title, price * 0.5 / 0.9, author, ISBN);
        this.runningTime = runningTime;
    }

    public double getPrice() { //Audiobook gives a 10% discount
        return super.getPrice();
    }

    public double getRunningTime() {
        return this.runningTime;
    }

    @Override
    public String toString() { //Appends runningtime to the end of the
        // toString in Book
        String result = String.format(" | RunningTime: %.2f", getRunningTime());
        return super.toString() + result;
    }
}
