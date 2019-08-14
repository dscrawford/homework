//Made by Daniel Crawford on 3/3/2018(dsc160130)
//Random Binary Tree(BT) will utilize Math.random to determine where nodes
// will go
public class RandomBT {
    AVLNode root;
    int size;
    RandomBT() {
        root = null;
        size = 0;
    }
    //For insertion without AVLNode already made
    public void insert(Book book, String key) {
        AVLNode newnode = new AVLNode(book, key);
        insert(newnode);
    }
    //For insertion with AVLNode
    public void insert(AVLNode newnode) {
        root = insert(newnode, root);
    }
    //Insert node into tree.
    private AVLNode insert(AVLNode newnode, AVLNode current) {
        if (current == null) {
            current = newnode;
        }
        else if (newnode.equals(current)) {
            System.out.println("ERROR: Can not insert ISBN's of the same " +
                    "value(ISBN: " + current.key + ")");
        }
        else {
            //If side == 0, chose left, otherwise chose right
            int side = (int)( (Math.random()*2) - 0.01);
            if (side == 0) {
                current.left = insert(newnode, current.left);
            }
            else {
                current.right = insert(newnode, current.right);
            }
        }
        current.findHeight();
        return current;
    }

    public void peak() {
        if (root != null)
            root.book.display();
        else
            System.out.println("The tree is empty.");
    }
    //        current.book.display();

    public void inorderFindErrors() {
        AVLNode temp = root;
        inorderFindErrors(temp, temp);
    }
    private void inorderFindErrors(AVLNode current, AVLNode parent) {
        if (current == null)
            return;
        inorderFindErrors(current.left,parent);
        findErrors(current, parent);
        inorderFindErrors(current.right,parent);
    }

    public void inorder() {
        AVLNode temp = root;
        inorder(temp);
    }
    private void inorder(AVLNode temp) {
        if (temp != null) {
            inorder(temp.left);
            temp.book.display();
            inorder(temp.right);
        }
    }

    private void findErrors(AVLNode current, AVLNode parent) {
        int balance = current.getBalance();
        //If out of balance, tell user
        if (balance < -1 || balance > 1) {
            System.out.println("AVL Problem: ISBN " + current.key + " breaks " +
                    "AVL balance condition(balance = " + balance + ").");
        }
        //If greater than parent on left tree, tell user
        if (current == parent.left) {
            if (current.compare(parent) > 0) {
                System.out.println("BST Probem: ISBN " + current.key + " " +
                        "breaks BST balance condition(left child ISBN: " +
                        current.key + " > " + "parent ISBN: " + parent.key +
                        ")");
            }
        }
        //If lesser than parent on right tree, tell user
        if (current == parent.right) {
            if (current.compare(parent) < 0) {
                System.out.println("BST Probem: ISBN " + current.key + " " +
                        "breaks BST balance condition(right child ISBN: " +
                        current.key + " < " + "parent ISBN: " + parent.key +
                        ")");
            }
        }
    }
}