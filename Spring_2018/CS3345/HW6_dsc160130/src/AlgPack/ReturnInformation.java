package AlgPack;

//Simply holds comparisons and movements so it can be returned.
public class ReturnInformation {
    long comparisons;
    long movements;

    ReturnInformation() {
        comparisons = 0;
        movements = 0;
    }

    public long getComparisons() {
        return this.comparisons;
    }

    public long getMovements() {
        return this.movements;
    }
}