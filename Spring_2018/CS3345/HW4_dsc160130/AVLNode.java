//Made by Daniel Crawford on 3/3/2018(dsc160130)
class AVLNode {
    String key;
    public Book book = new Book("null", "null");
    int height;
    protected AVLNode left;
    protected AVLNode right;

    AVLNode(Book book, String key) {
        this.book = book;
        this.key = key;
        this.right = null;
        this.left = null;
        height = 0;
    }

    protected void findHeight() {
        //If node is a leaf, height is 0
        if (left == null && right == null)
            this.height = 0;
            //If no left subtree
        else if (left == null && right != null)
            this.height = right.height + 1;
            //If no right subtree
        else if (left != null && right == null)
            this.height = left.height + 1;
        //If node has children, height is from highest child
        else if (left.height > right.height)
            this.height = left.height + 1;
        else
            this.height = right.height + 1;
    }

    //Balance = leftheight - rightheight
    protected int getBalance() {
        int right_height, left_height;

        //If no right node, height is -1
        if (right == null)
            right_height = -1;
        else
            right_height = right.height;

        //If no left node, height is -1
        if (left == null)
            left_height = -1;
        else
            left_height = left.height;

        return left_height - right_height;
    }

    //If lesser than 0, passed argument is greater,
    // if equal to 0, both nodes are equal,
    // if greater than 0, this instance of node is greater
    public int compare(AVLNode node) {
        return Integer.parseInt(this.key) - Integer.parseInt(node.key);
    }
}
