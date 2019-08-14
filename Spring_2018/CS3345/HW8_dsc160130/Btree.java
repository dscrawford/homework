import java.util.ArrayList;


class Btree<T extends Comparable<T>> {
    class Node {
        //List of data in current node
        private ArrayList<T> keys;
        //Children held by this data
        private ArrayList<Node> children;
        private int t; //Maximum amount of degrees allowed
        private int size;
        private boolean isLeaf;

        private Node(int t, boolean isLeaf) {
            this.t = t;
            this.size = 0;
            this.isLeaf = isLeaf;

            keys = new ArrayList<>(t);
            children = new ArrayList<>(t);
        }

        private void traverse() {
            int i;
            if (!keys.isEmpty()) {
                for (i = 0; i < size; i++) {
                    //If the current node isn't a leaf, then go to its children
                    if (!isLeaf) {
                        System.out.println("Traversing to leaf " + (i + 1));
                        children.get(i).traverse();
                    }
                    if(!isLeaf)
                        System.out.println("Displaying inside internal node");
                    System.out.println(keys.get(i));
                }

                //Print rightmost child tree
                if (!isLeaf) {
                    System.out.println("Traversing to rightmost leaf");
                    children.get(i).traverse();
                }
            }
        }

        //Print the subtree with the last child
        public Node search(T k) {
            int i = 0;
            while (i < size && keys.get(i).compareTo(k) < 0) {
                i++;
            }

            if (keys.get(i) == k)
                return this;

            if (isLeaf)
                return null;

            return children.get(i).search(k);
        }

        //Node k MUST be full if this function is called.
        private void split(int i, Node k) {
            int x = t / 2; //The middle of the Node
            Node temp = new Node(k.t, k.isLeaf);
            temp.size = x - 1;

            //Add each node before the half point to the key values.
            for (int j = 0; j < x - 1; j++) {
                temp.keys.add(k.keys.get(j+x));
            }

            //If k has children, find them and put them inside the new node.
            if (!k.isLeaf) {
                for (int j = 0; j < x; j++) {
                    temp.children.add(k.children.get(j+x));
                }
            }

            k.size = x - 1;


            children.add(i+1, temp);

            keys.add(i, k.keys.get(x - 1));

            //Remove any leftover elements in k
            for (int j = 0; j < x - 1; j++) {
                k.keys.remove(j+x);
            }
            k.keys.remove(x-1);


            size++; //increase the size
        }

        private void insertNotFull(T k) {
            //Start at rightmost element
            int i = size - 1;

            if (isLeaf) {
                //if keys[size - 1] <= k, then append the node
                if (keys.get(i).compareTo(k) < 0) {
                    keys.add(k);
                }
                //Otherwise, find where it should be placed.
                else {
                    while (i >= 0) {
                        if (keys.get(i).compareTo(k) >= 0) {
                            keys.add(i,k);
                            break;
                        }
                        i--;
                    }
                }
                size++;
            }
            else {
                //If the node is not a leaf, keep looking until it finds one
                //Find the child with the new key
                //while i >= 0 and keys[i] > k
                while (i >= 0 && keys.get(i).compareTo(k) >= 0) {
                    i--;
                }

                if (children.get(i+1).size == t - 1) {
                    split(i+1, children.get(i+1));

                    //If keys[i+1] < k, there will be two children after this
                    // split.
                    if(keys.get(i+1).compareTo(k) < 0)
                        i++;
                }
                children.get(i+1).insertNotFull(k);
            }

        }
    }

    private Node root;
    private int t; //Maximum degree

    public Btree(int t) {
        root = null;
        this.t = t;
    }

    public void traverse() {
        if (root != null) {
            Node current = root;
            current.traverse();
        }
    }

    public Node search(T k) {
        if (root == null)
            return null;
        else {
            Node current = root;
            return current.search(k);
        }
    }

    public void insert(T k) {
        //If the tree is empty, simply insert
        if (root == null) {
            root = new Node(t, true);
            root.keys.add(k);
            root.size = 1;
        }
        //Otherwise tree must not be empty.
        else {
            //If the size is full
            if (root.size == t - 1) {
                Node temp = new Node(t, false);
                temp.children.add(root);

                temp.split(0, root);


                int i = 0;
                //Try to find child keys[i] < k, following BST structure
                if (temp.keys.get(i).compareTo(k) < 0)
                    i++;
                temp.children.get(i).insertNotFull(k);

                root = temp;
            }
            else //If root is not full, simply insert new value into it.
                root.insertNotFull(k);
        }
    }


}
