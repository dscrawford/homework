public class AVL  {
    AVLNode root;
    public int size;

    AVL() {
        this.root = null;
        this.size = 0;
    }

    //For insertion which AVLNode object has not been made
    public void insert(Book book, String key) {
        AVLNode newnode = new AVLNode(book, key);
        insert(newnode);
    }

    //For insertion where AVLNode object has been made
    public void insert(AVLNode newnode) {
        root = insert(newnode, root);
    }

    //Actual insertion
    private AVLNode insert(AVLNode newnode, AVLNode current) {
        boolean madeChange = false;
        if (current == null) {
            madeChange = true;
            current = newnode;
        }
        //If newnode is lesser than the current node
        else if (newnode.compare(current) < 0) {
            current.left = insert(newnode, current.left);
        }
        //If newnode is greater than the newer node
        else if (newnode.compare(current) > 0) {
            current.right = insert(newnode, current.right);
        }
        //Otherwise, ISBN is already present and should not be added
        else {
            System.out.println("ERROR: ISBN " + newnode.key + " already " +
                    "exists!");
        }

        if (madeChange)
            this.size++;


        current.findHeight();

        int balance = current.getBalance();
        if (balance < -1 || balance > 1) {//If tree is imbalanced, rotate
            current = AVLBalance(current, newnode, balance);
        }

        return current;
    }
    //Rotate tree assuming the balance has already been determined to not fit
    // AVL standards
    private AVLNode AVLBalance(AVLNode current, AVLNode newnode, int balance) {
        //If the imbalance if coming from the right side
        if (balance < -1) {
            //Right right case(when newnode is greater than the right key)
            if (newnode.compare(current.right) > 0) {
                System.out.println("Imbalance occurred at inserting ISBN " +
                        current.key + "; fixed in right right rotation.");
                current = leftRotate(current);
            }
            //right left case
            else {
                System.out.println("Imbalance occurred at inserting ISBN " +
                        current.key + "; fixed in right left rotation.");
                current.right = rightRotate(current.right);
                current = leftRotate(current);
            }
        }//Right left can(when newnode is lesser than the right key)
        //If the imbalance if coming from the left side
        else {
            //left left rotation(when newnode is lesser than the left key)
            if (newnode.compare(current.left) < 0) {
                System.out.println("Imbalance occurred at inserting ISBN " +
                        current.key + "; fixed in left left rotation.");
                current = rightRotate(current);
            }
            //left right rotation(when newnode is greater than the left key)
            else {
                System.out.println("Imbalance occurred at inserting ISBN " +
                        current.key + "; fixed in left right rotation.");
                current.left = leftRotate(current);
                current = rightRotate(current);
            }

        }
        return current;
    }
    /* Left rotation looks like below diagram
           2             1
            \    -->    / \
             1         0   2
              \
               0
     */
    private AVLNode leftRotate(AVLNode node1) {
        AVLNode node2 = node1.right;
        //left children of right node will be adopted by node1
        AVLNode adoptnode = node2.left;

        //Rotate in left direction

        node1.right = adoptnode;
        node2.left = node1;

        node1.findHeight();
        node2.findHeight();

        return node2;
    }

    /* Right rotation looks like below diagram
           2            1
          /     -->    / \
         1            0   2
        /
       0
     */
    private AVLNode rightRotate(AVLNode node1) {
        AVLNode node2 = node1.left;
        //right children of left node will be adopted by node1
        AVLNode adoptnode = node2.right;


        //Rotate in right direction
        node2.right = node1;
        node1.left = adoptnode;

        node1.findHeight();
        node2.findHeight();

        return node2;
    }
    public void peak() {
        if (root != null)
            root.book.display();
        else
            System.out.println("The tree is empty.");
    }

    public void preorder() {
        AVLNode temp = root;
        preorder(temp);
    }
    private void preorder(AVLNode temp) {
        if (temp != null) {
            temp.book.display();
            preorder(temp.left);
            preorder(temp.right);
        }
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

    public void postorder() {
        AVLNode temp = root;
        preorder(temp);
    }
    private void postorder(AVLNode temp) {
        if (temp != null) {
            postorder(temp.left);
            postorder(temp.right);
            temp.book.display();

        }
    }
}
