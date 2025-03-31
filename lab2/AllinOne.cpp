#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <random>
using namespace std;

// Node structure for the binary search tree
class TreeNode {
    public:
        double value; // Value of the node
        TreeNode* left; // Pointer to the left child
        TreeNode* right; // Pointer to the right child
    
        // Constructor to initialize a node
        TreeNode(double val) : value(val), left(nullptr), right(nullptr) {}
    };
    
// Binary Search Tree class
class BinarySearchTree {
private:
    TreeNode* root; // Root of the tree

    // Helper function to insert a value into the tree
    TreeNode* insert(TreeNode* node, double value) {
        if (node == nullptr) {
            return new TreeNode(value); // Create a new node if the current node is null
        }
        if (value <= node->value) {
            node->left = insert(node->left, value); // Insert into the left subtree
        } else {
            node->right = insert(node->right, value); // Insert into the right subtree
        }
        return node;
    }

    // Helper function to search for a value in the tree
    bool search(TreeNode* node, double value) {
        if (node == nullptr) {
            return false; // Value not found
        }
        if (node->value == value) {
            return true; // Value found
        }
        if (value < node->value) {
            return search(node->left, value); // Search in the left subtree
        } else {
            return search(node->right, value); // Search in the right subtree
        }
    }

    // Helper function to find the minimum value node in a subtree
    TreeNode* findMin(TreeNode* node) {
        while (node->left != nullptr) {
            node = node->left; // Move to the leftmost node
        }
        return node;
    }

    // Helper function to delete a value from the tree
    TreeNode* remove(TreeNode* node, double value) {
        if (node == nullptr) {
            return nullptr; // Value not found
        }
        if (value < node->value) {
            node->left = remove(node->left, value); // Delete from the left subtree
        } else if (value > node->value) {
            node->right = remove(node->right, value); // Delete from the right subtree
        } else {
            // Node with only one child or no child
            if (node->left == nullptr) {
                TreeNode* temp = node->right;
                delete node;
                return temp;
            } else if (node->right == nullptr) {
                TreeNode* temp = node->left;
                delete node;
                return temp;
            }

            // Node with two children: Get the inorder successor (smallest in the right subtree)
            TreeNode* temp = findMin(node->right);
            node->value = temp->value; // Copy the inorder successor's value to this node
            node->right = remove(node->right, temp->value); // Delete the inorder successor
        }
        return node;
    }

    // Helper function to print the tree in-order
    void inOrder(TreeNode* node) {
        if (node == nullptr) {
            return;
        }
        inOrder(node->left); // Visit left subtree
        cout << node->value << " "; // Print node value
        inOrder(node->right); // Visit right subtree
    }

public:
    // Constructor to initialize the tree
    BinarySearchTree() : root(nullptr) {}

    // Method to insert a value into the tree
    void insert(double value) {
        root = insert(root, value);
    }

    // Method to search for a value in the tree
    bool search(double value) {
        return search(root, value);
    }

    // Method to delete a value from the tree
    void remove(double value) {
        root = remove(root, value);
    }

    // Method to print the tree in-order
    void printInOrder() {
        inOrder(root);
        cout << endl;
    }
};

// AVL Tree Node
class AVLNode {
    public:
        double key;
        int height;
        AVLNode* left;
        AVLNode* right;
    
        AVLNode(double value) : key(value), height(1), left(nullptr), right(nullptr) {}
    };
    
// AVL Tree Class
class AVLTree {
private:
    AVLNode* root;

    // Get the height of a node
    int getHeight(AVLNode* node) {
        return node ? node->height : 0;
    }

    // Calculate balance factor of a node
    int getBalanceFactor(AVLNode* node) {
        return node ? getHeight(node->left) - getHeight(node->right) : 0;
    }

    // Right rotation
    AVLNode* rotateRight(AVLNode* y) {
        AVLNode* x = y->left;
        AVLNode* T2 = x->right;

        x->right = y;
        y->left = T2;

        y->height = max(getHeight(y->left), getHeight(y->right)) + 1;
        x->height = max(getHeight(x->left), getHeight(x->right)) + 1;

        return x;
    }

    // Left rotation
    AVLNode* rotateLeft(AVLNode* x) {
        AVLNode* y = x->right;
        AVLNode* T2 = y->left;

        y->left = x;
        x->right = T2;

        x->height = max(getHeight(x->left), getHeight(x->right)) + 1;
        y->height = max(getHeight(y->left), getHeight(y->right)) + 1;

        return y;
    }

    // Insert a key into the AVL tree
    AVLNode* insert(AVLNode* node, double key) {
        if (!node) return new AVLNode(key);

        if (key <= node->key) {
            node->left = insert(node->left, key);
        } else {
            node->right = insert(node->right, key);
        }

        node->height = 1 + max(getHeight(node->left), getHeight(node->right));

        int balance = getBalanceFactor(node);

        // Left Left Case
        if (balance > 1 && key <= node->left->key) {
            return rotateRight(node);
        }

        // Right Right Case
        if (balance < -1 && key > node->right->key) {
            return rotateLeft(node);
        }

        // Left Right Case
        if (balance > 1 && key > node->left->key) {
            node->left = rotateLeft(node->left);
            return rotateRight(node);
        }

        // Right Left Case
        if (balance < -1 && key <= node->right->key) {
            node->right = rotateRight(node->right);
            return rotateLeft(node);
        }

        return node;
    }

    // Find the node with the smallest key
    AVLNode* findMin(AVLNode* node) {
        while (node->left) {
            node = node->left;
        }
        return node;
    }

    // Delete a key from the AVL tree
    AVLNode* deleteNode(AVLNode* node, double key) {
        if (!node) return node;

        if (key < node->key) {
            node->left = deleteNode(node->left, key);
        } else if (key > node->key) {
            node->right = deleteNode(node->right, key);
        } else {
            if (!node->left || !node->right) {
                AVLNode* temp = node->left ? node->left : node->right;
                delete node;
                return temp;
            } else {
                AVLNode* temp = findMin(node->right);
                node->key = temp->key;
                node->right = deleteNode(node->right, temp->key);
            }
        }

        node->height = 1 + max(getHeight(node->left), getHeight(node->right));

        int balance = getBalanceFactor(node);

        // Left Left Case
        if (balance > 1 && getBalanceFactor(node->left) >= 0) {
            return rotateRight(node);
        }

        // Left Right Case
        if (balance > 1 && getBalanceFactor(node->left) < 0) {
            node->left = rotateLeft(node->left);
            return rotateRight(node);
        }

        // Right Right Case
        if (balance < -1 && getBalanceFactor(node->right) <= 0) {
            return rotateLeft(node);
        }

        // Right Left Case
        if (balance < -1 && getBalanceFactor(node->right) > 0) {
            node->right = rotateRight(node->right);
            return rotateLeft(node);
        }

        return node;
    }

    // Search for a key in the AVL tree
    bool search(AVLNode* node, double key) {
        if (!node) return false;

        if (key == node->key) return true;
        if (key < node->key) return search(node->left, key);
        return search(node->right, key);
    }

    // In-order traversal
    void inOrder(AVLNode* node) {
        if (node) {
            inOrder(node->left);
            cout << node->key << " ";
            inOrder(node->right);
        }
    }

public:
    AVLTree() : root(nullptr) {}

    void insert(double key) {
        root = insert(root, key);
    }

    void deleteKey(double key) {
        root = deleteNode(root, key);
    }

    bool search(double key) {
        return search(root, key);
    }

    void display() {
        inOrder(root);
        cout << endl;
    }
};

// Define color constants
enum Color { RED, BLACK };

// Node structure for Red-Black Tree
struct RBNode {
    double key;          // The key (value) stored in the node
    int count;           // Count of duplicate keys
    Color color;         // Node color
    RBNode *left;        // Pointer to left child
    RBNode *right;       // Pointer to right child
    RBNode *parent;      // Pointer to parent

    RBNode(double key, RBNode* nil)
        : key(key), count(1), color(RED), left(nil), right(nil), parent(nil) {}
};

class RBTree {
public:
    RBTree() {
        // Create the sentinel nil node, which is black.
        nil = new RBNode(0, nullptr);
        nil->color = BLACK;
        nil->left = nil->right = nil->parent = nil;
        root = nil;
    }

    ~RBTree() {
        destroyTree(root);
        delete nil;
    }

    // Insert key into red-black tree; allow duplicate via count increment.
    void insert(double key) {
        RBNode* z = new RBNode(key, nil);
        RBNode* y = nil;
        RBNode* x = root;
        // Standard BST insertion to find the correct location.
        while (x != nil) {
            y = x;
            if (key == x->key) {
                // Duplicate found: increment count and delete the newly allocated node.
                x->count++;
                delete z;
                return;
            } else if (key < x->key) {
                x = x->left;
            } else {
                x = x->right;
            }
        }
        z->parent = y;
        if (y == nil) {
            root = z;
        } else if (key < y->key) {
            y->left = z;
        } else {
            y->right = z;
        }
        z->color = RED;
        insertFixup(z);
    }

    // Delete one instance of the key from the tree.
    void deleteKey(double key) {
        RBNode* z = search(root, key);
        if (z == nil) {
            cout << "Key " << key << " not found in the tree." << endl;
            return;
        }
        // If there are duplicates, just decrement the count.
        if (z->count > 1) {
            z->count--;
            return;
        }
        RBNode* y = z;
        Color yOriginalColor = y->color;
        RBNode* x;
        if (z->left == nil) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == nil) {
            x = z->left;
            transplant(z, z->left);
        } else {
            y = minimum(z->right);
            yOriginalColor = y->color;
            x = y->right;
            if (y->parent == z) {
                x->parent = y;
            } else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }
            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }
        delete z;
        if (yOriginalColor == BLACK) {
            deleteFixup(x);
        }
    }

    // Search for a node with the given key.
    // Returns pointer to the node if found; otherwise returns nil.
    RBNode* search(double key) {
        return search(root, key);
    }

    // Inorder traversal of the tree (for testing purposes).
    void inorder() {
        inorder(root);
    }

private:
    RBNode* root;
    RBNode* nil; // Sentinel node

    // Recursively destroy the tree nodes.
    void destroyTree(RBNode* node) {
        if (node != nil) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }

    // Left rotate around node x.
    void leftRotate(RBNode* x) {
        RBNode* y = x->right;
        x->right = y->left;
        if (y->left != nil) {
            y->left->parent = x;
        }
        y->parent = x->parent;
        if (x->parent == nil) {
            root = y;
        } else if (x == x->parent->left) {
            x->parent->left = y;
        } else {
            x->parent->right = y;
        }
        y->left = x;
        x->parent = y;
    }

    // Right rotate around node y.
    void rightRotate(RBNode* y) {
        RBNode* x = y->left;
        y->left = x->right;
        if (x->right != nil) {
            x->right->parent = y;
        }
        x->parent = y->parent;
        if (y->parent == nil) {
            root = x;
        } else if (y == y->parent->right) {
            y->parent->right = x;
        } else {
            y->parent->left = x;
        }
        x->right = y;
        y->parent = x;
    }

    // Fix-up the tree after insertion to maintain red-black properties.
    void insertFixup(RBNode* z) {
        while (z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) {
                RBNode* y = z->parent->parent->right; // uncle
                if (y->color == RED) {
                    // Case 1: uncle is red
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->right) {
                        // Case 2: z is a right child
                        z = z->parent;
                        leftRotate(z);
                    }
                    // Case 3: z is a left child
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    rightRotate(z->parent->parent);
                }
            } else { // Same as above "if" clause with "right" and "left" exchanged.
                RBNode* y = z->parent->parent->left; // uncle
                if (y->color == RED) {
                    // Case 1: uncle is red
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->left) {
                        // Case 2: z is a left child
                        z = z->parent;
                        rightRotate(z);
                    }
                    // Case 3: z is a right child
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    leftRotate(z->parent->parent);
                }
            }
        }
        root->color = BLACK;
    }

    // Transplant subtree u with subtree v.
    void transplant(RBNode* u, RBNode* v) {
        if (u->parent == nil) {
            root = v;
        } else if (u == u->parent->left) {
            u->parent->left = v;
        } else {
            u->parent->right = v;
        }
        v->parent = u->parent;
    }

    // Fix-up the tree after deletion to maintain red-black properties.
    void deleteFixup(RBNode* x) {
        while (x != root && x->color == BLACK) {
            if (x == x->parent->left) {
                RBNode* w = x->parent->right; // sibling
                if (w->color == RED) {
                    // Case 1
                    w->color = BLACK;
                    x->parent->color = RED;
                    leftRotate(x->parent);
                    w = x->parent->right;
                }
                if (w->left->color == BLACK && w->right->color == BLACK) {
                    // Case 2
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->right->color == BLACK) {
                        // Case 3
                        w->left->color = BLACK;
                        w->color = RED;
                        rightRotate(w);
                        w = x->parent->right;
                    }
                    // Case 4
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->right->color = BLACK;
                    leftRotate(x->parent);
                    x = root;
                }
            } else {
                // Mirror image of above code with "left" and "right" exchanged.
                RBNode* w = x->parent->left;
                if (w->color == RED) {
                    // Case 1
                    w->color = BLACK;
                    x->parent->color = RED;
                    rightRotate(x->parent);
                    w = x->parent->left;
                }
                if (w->right->color == BLACK && w->left->color == BLACK) {
                    // Case 2
                    w->color = RED;
                    x = x->parent;
                } else {
                    if (w->left->color == BLACK) {
                        // Case 3
                        w->right->color = BLACK;
                        w->color = RED;
                        leftRotate(w);
                        w = x->parent->left;
                    }
                    // Case 4
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->left->color = BLACK;
                    rightRotate(x->parent);
                    x = root;
                }
            }
        }
        x->color = BLACK;
    }

    // Return the minimum node in the subtree rooted at node.
    RBNode* minimum(RBNode* node) {
        while (node->left != nil) {
            node = node->left;
        }
        return node;
    }

    // Recursive helper function for searching the tree.
    RBNode* search(RBNode* node, double key) {
        if (node == nil || key == node->key)
            return node;
        if (key < node->key)
            return search(node->left, key);
        else
            return search(node->right, key);
    }

    // Inorder traversal helper function.
    void inorder(RBNode* node) {
        if (node != nil) {
            inorder(node->left);
            cout << "Key: " << node->key << " Count: " << node->count
                 << " Color: " << (node->color == RED ? "RED" : "BLACK") << endl;
            inorder(node->right);
        }
    }
};

// Define the order of the B-Tree
const int ORDER = 5;

// B-Tree Node class
class BTreeNode {
public:
    vector<double> keys; // Keys in the node
    vector<BTreeNode*> children; // Child pointers
    bool isLeaf; // True if the node is a leaf

    BTreeNode(bool leaf) : isLeaf(leaf) {}

    // Insert a key into a non-full node
    void insertNonFull(double key);

    // Split a child node
    void splitChild(int i, BTreeNode* child);

    // Search for a key in the subtree rooted at this node
    BTreeNode* search(double key);

    // Find the index of the first key greater than or equal to key
    int findKey(double key);

    // Remove a key from the subtree rooted at this node
    void remove(double key);

    // Remove a key from a leaf node
    void removeFromLeaf(int idx);

    // Remove a key from a non-leaf node
    void removeFromNonLeaf(int idx);

    // Get the predecessor of a key
    double getPredecessor(int idx);

    // Get the successor of a key
    double getSuccessor(int idx);

    // Fill a child node that has less than the minimum number of keys
    void fill(int idx);

    // Borrow a key from the previous sibling
    void borrowFromPrev(int idx);

    // Borrow a key from the next sibling
    void borrowFromNext(int idx);

    // Merge a child with its sibling
    void merge(int idx);

    friend class BTree;
};

// B-Tree class
class BTree {
private:
    BTreeNode* root; // Pointer to the root node
public:
    BTree() : root(nullptr) {}

    // Search for a key in the B-Tree
    BTreeNode* search(double key) {
        return root == nullptr ? nullptr : root->search(key);
    }

    // Insert a key into the B-Tree
    void insert(double key);

    // Remove a key from the B-Tree
    void remove(double key);

    // Print the B-Tree (for debugging)
    void print() {
        if (root != nullptr) printTree(root, 0);
    }

private:
    void printTree(BTreeNode* node, int depth);
};

// Insert a key into the B-Tree
void BTree::insert(double key) {
    if (root == nullptr) {
        root = new BTreeNode(true);
        root->keys.push_back(key);
    } else {
        if (root->keys.size() == ORDER - 1) {
            BTreeNode* newRoot = new BTreeNode(false);
            newRoot->children.push_back(root);
            newRoot->splitChild(0, root);
            int i = (newRoot->keys[0] < key) ? 1 : 0;
            newRoot->children[i]->insertNonFull(key);
            root = newRoot;
        } else {
            root->insertNonFull(key);
        }
    }
}

// Insert a key into a non-full node
void BTreeNode::insertNonFull(double key) {
    int i = keys.size() - 1;
    if (isLeaf) {
        keys.push_back(0);
        while (i >= 0 && keys[i] > key) {
            keys[i + 1] = keys[i];
            i--;
        }
        keys[i + 1] = key;
    } else {
        while (i >= 0 && keys[i] > key) i--;
        if (children[i + 1]->keys.size() == ORDER - 1) {
            splitChild(i + 1, children[i + 1]);
            if (keys[i + 1] < key) i++;
        }
        children[i + 1]->insertNonFull(key);
    }
}

// Split a child node
void BTreeNode::splitChild(int i, BTreeNode* child) {
    BTreeNode* newChild = new BTreeNode(child->isLeaf);
    int mid = ORDER / 2;
    newChild->keys.assign(child->keys.begin() + mid + 1, child->keys.end());
    child->keys.resize(mid);

    if (!child->isLeaf) {
        newChild->children.assign(child->children.begin() + mid + 1, child->children.end());
        child->children.resize(mid + 1);
    }

    children.insert(children.begin() + i + 1, newChild);
    keys.insert(keys.begin() + i, child->keys[mid]);
}

// Search for a key in the subtree rooted at this node
BTreeNode* BTreeNode::search(double key) {
    int i = 0;
    while (i < keys.size() && key > keys[i]) i++;
    if (i < keys.size() && keys[i] == key) return this;
    return isLeaf ? nullptr : children[i]->search(key);
}

// Remove a key from the B-Tree
void BTree::remove(double key) {
    if (!root) return;
    root->remove(key);
    if (root->keys.empty()) {
        BTreeNode* oldRoot = root;
        root = root->isLeaf ? nullptr : root->children[0];
        delete oldRoot;
    }
}

// Remove a key from the subtree rooted at this node
void BTreeNode::remove(double key) {
    int idx = findKey(key);
    if (idx < keys.size() && keys[idx] == key) {
        if (isLeaf) removeFromLeaf(idx);
        else removeFromNonLeaf(idx);
    } else {
        if (isLeaf) return;
        bool flag = (idx == keys.size());
        if (children[idx]->keys.size() < (ORDER + 1) / 2) fill(idx);
        if (flag && idx > keys.size()) children[idx - 1]->remove(key);
        else children[idx]->remove(key);
    }
}

// Find the index of the first key greater than or equal to key
int BTreeNode::findKey(double key) {
    int idx = 0;
    while (idx < keys.size() && keys[idx] < key) ++idx;
    return idx;
}

// Remove a key from a leaf node
void BTreeNode::removeFromLeaf(int idx) {
    keys.erase(keys.begin() + idx);
}

// Remove a key from a non-leaf node
void BTreeNode::removeFromNonLeaf(int idx) {
    double key = keys[idx];
    if (children[idx]->keys.size() >= (ORDER + 1) / 2) {
        double pred = getPredecessor(idx);
        keys[idx] = pred;
        children[idx]->remove(pred);
    } else if (children[idx + 1]->keys.size() >= (ORDER + 1) / 2) {
        double succ = getSuccessor(idx);
        keys[idx] = succ;
        children[idx + 1]->remove(succ);
    } else {
        merge(idx);
        children[idx]->remove(key);
    }
}

// Get the predecessor of a key
double BTreeNode::getPredecessor(int idx) {
    BTreeNode* cur = children[idx];
    while (!cur->isLeaf) cur = cur->children.back();
    return cur->keys.back();
}

// Get the successor of a key
double BTreeNode::getSuccessor(int idx) {
    BTreeNode* cur = children[idx + 1];
    while (!cur->isLeaf) cur = cur->children.front();
    return cur->keys.front();
}

// Fill a child node that has less than the minimum number of keys
void BTreeNode::fill(int idx) {
    if (idx != 0 && children[idx - 1]->keys.size() >= (ORDER + 1) / 2) {
        borrowFromPrev(idx);
    } else if (idx != keys.size() && children[idx + 1]->keys.size() >= (ORDER + 1) / 2) {
        borrowFromNext(idx);
    } else {
        if (idx != keys.size()) merge(idx);
        else merge(idx - 1);
    }
}

// Borrow a key from the previous sibling
void BTreeNode::borrowFromPrev(int idx) {
    BTreeNode* child = children[idx];
    BTreeNode* sibling = children[idx - 1];

    child->keys.insert(child->keys.begin(), keys[idx - 1]);
    if (!child->isLeaf) {
        child->children.insert(child->children.begin(), sibling->children.back());
        sibling->children.pop_back();
    }
    keys[idx - 1] = sibling->keys.back();
    sibling->keys.pop_back();
}

// Borrow a key from the next sibling
void BTreeNode::borrowFromNext(int idx) {
    BTreeNode* child = children[idx];
    BTreeNode* sibling = children[idx + 1];

    child->keys.push_back(keys[idx]);
    if (!child->isLeaf) {
        child->children.push_back(sibling->children.front());
        sibling->children.erase(sibling->children.begin());
    }
    keys[idx] = sibling->keys.front();
    sibling->keys.erase(sibling->keys.begin());
}

// Merge a child with its sibling
void BTreeNode::merge(int idx) {
    BTreeNode* child = children[idx];
    BTreeNode* sibling = children[idx + 1];

    child->keys.push_back(keys[idx]);
    child->keys.insert(child->keys.end(), sibling->keys.begin(), sibling->keys.end());
    if (!child->isLeaf) {
        child->children.insert(child->children.end(), sibling->children.begin(), sibling->children.end());
    }
    keys.erase(keys.begin() + idx);
    children.erase(children.begin() + idx + 1);
    delete sibling;
}

// Print the B-Tree (for debugging)
void BTree::printTree(BTreeNode* node, int depth) {
    for (int i = 0; i < depth; ++i) cout << "  ";
    for (double key : node->keys) cout << key << " ";
    cout << endl;
    if (!node->isLeaf) {
        for (BTreeNode* child : node->children) {
            printTree(child, depth + 1);
        }
    }
}

// Function to test the performance of the trees
void testPerformance(int n, double searchKey, double deleteKey) {
    // Generate random integers
    vector<double> testData;
    testData.reserve(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 10000);
    for (int i = 0; i < n; ++i) {
        testData.push_back(dis(gen));
    }
    if (n < 1000) {
        cout << "Test data: ";
        for (double val : testData) {
            cout << val << " ";
        }
    }
    // Measure time to insert elements
    cout << "\nTesting with n = " << n << " elements:\n";

    auto start = chrono::high_resolution_clock::now();
    BinarySearchTree bst;
    for (double val : testData) bst.insert(val);
    auto end = chrono::high_resolution_clock::now();
    cout << "  Binary Search Tree insertion time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    AVLTree avl;
    for (double val : testData) avl.insert(val);
    end = chrono::high_resolution_clock::now();
    cout << "  AVL Tree insertion time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    RBTree rbt;
    for (double val : testData) rbt.insert(val);
    end = chrono::high_resolution_clock::now();
    cout << "  Red-Black Tree insertion time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    BTree btree;
    for (double val : testData) btree.insert(val);
    end = chrono::high_resolution_clock::now();
    cout << "  B-Tree insertion time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

    // Measure time to search for an element
    cout << "\nSearching for key " << searchKey << ":\n";

    start = chrono::high_resolution_clock::now();
    bst.search(searchKey);
    end = chrono::high_resolution_clock::now();
    cout << "  Binary Search Tree search time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    start = chrono::high_resolution_clock::now();
    avl.search(searchKey);
    end = chrono::high_resolution_clock::now();
    cout << "  AVL Tree search time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    start = chrono::high_resolution_clock::now();
    rbt.search(searchKey);
    end = chrono::high_resolution_clock::now();
    cout << "  Red-Black Tree search time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    start = chrono::high_resolution_clock::now();
    btree.search(searchKey);
    end = chrono::high_resolution_clock::now();
    cout << "  B-Tree search time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    // Measure time to delete an element
    cout << "\nDeleting key " << deleteKey << ":\n";

    start = chrono::high_resolution_clock::now();
    bst.remove(deleteKey);
    end = chrono::high_resolution_clock::now();
    cout << "  Binary Search Tree deletion time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    start = chrono::high_resolution_clock::now();
    avl.deleteKey(deleteKey);
    end = chrono::high_resolution_clock::now();
    cout << "  AVL Tree deletion time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    start = chrono::high_resolution_clock::now();
    rbt.deleteKey(deleteKey);
    end = chrono::high_resolution_clock::now();
    cout << "  Red-Black Tree deletion time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";

    start = chrono::high_resolution_clock::now();
    btree.remove(deleteKey);
    end = chrono::high_resolution_clock::now();
    cout << "  B-Tree deletion time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us\n";
}

int main() {
    // Initialize the trees
    BinarySearchTree bst;
    AVLTree avl;
    RBTree rbt;
    BTree btree;

    // Test data
    vector<double> testData = {50, 30, 70, 20, 40, 60, 80, 10, 90, 100};

    cout << "Inserting elements into the trees...\n";
    for (double val : testData) {
        bst.insert(val);
        avl.insert(val);
        rbt.insert(val);
        btree.insert(val);
    }

    cout << "Binary Search Tree (In-Order): ";
    bst.printInOrder();

    cout << "AVL Tree (In-Order): ";
    avl.display();

    cout << "Red-Black Tree (In-Order):\n";
    rbt.inorder();

    cout << "B-Tree (Structure):\n";
    btree.print();

    cout << "\nSearching for elements in the trees...\n";
    vector<double> searchKeys = {30, 60, 100, 110};
    for (double key : searchKeys) {
        cout << "Searching for " << key << ":\n";
        cout << "  Binary Search Tree: " << (bst.search(key) ? "Found" : "Not Found") << endl;
        cout << "  AVL Tree: " << (avl.search(key) ? "Found" : "Not Found") << endl;
        cout << "  Red-Black Tree: " << (rbt.search(key) != nullptr ? "Found" : "Not Found") << endl;
        cout << "  B-Tree: " << (btree.search(key) != nullptr ? "Found" : "Not Found") << endl;
    }

    cout << "\nDeleting elements from the trees...\n";
    vector<double> deleteKeys = {50, 70, 10};
    for (double key : deleteKeys) {
        cout << "Deleting " << key << "...\n";
        bst.remove(key);
        avl.deleteKey(key);
        rbt.deleteKey(key);
        btree.remove(key);
    }

    cout << "\nAfter Deletion:\n";
    cout << "Binary Search Tree (In-Order): ";
    bst.printInOrder();

    cout << "AVL Tree (In-Order): ";
    avl.display();

    cout << "Red-Black Tree (In-Order):\n";
    rbt.inorder();

    cout << "B-Tree (Structure):\n";
    btree.print();

    // Run performance tests for different sizes
    testPerformance(1000, 10, 10);
    testPerformance(10000, 10, 10);
    testPerformance(20000, 10, 10);
    testPerformance(50000, 10, 10);
    testPerformance(100000, 10, 10);
    testPerformance(200000, 10, 10);
    testPerformance(500000, 10, 10);
    testPerformance(1000000, 10, 10);
    testPerformance(2000000, 10, 10);

    return 0;
}