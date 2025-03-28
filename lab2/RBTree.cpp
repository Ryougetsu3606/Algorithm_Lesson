// RBTree.cpp
// Implementation of a Red-Black Tree that supports duplicate elements.
// The tree supports insertion, deletion, and search operations.
// Duplicate elements are stored as a count in the node.
// English comments are provided for clarity.

#include <iostream>
#include <cstdlib>
using namespace std;

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

// Test functions
int main() {
    RBTree tree;

    // Insert elements into the tree (with duplicates)
    double arr[] = {10.5, 20.3, 10.5, 15.0, 25.7, 20.3, 5.2};
    int n = sizeof(arr) / sizeof(arr[0]);

    for (int i = 0; i < n; i++) {
        tree.insert(arr[i]);
    }

    cout << "Inorder traversal after insertions:" << endl;
    tree.inorder();

    // Search for a key
    double searchKey = 20.3;
    cout << "Searching for " << searchKey << ": " << (tree.search(searchKey)->key == searchKey ? "Found" : "Not Found") << endl;

    // Delete a key (one instance)
    tree.deleteKey(10.5);
    cout << "Inorder traversal after deleting ONE 10.5:" << endl;
    tree.inorder();

    // Delete the same key again (should remove the duplicate)
    tree.deleteKey(10.5);
    cout << "Inorder traversal after deleting ANOTHER 10.5:" << endl;
    tree.inorder();

    return 0;
}