#include <iostream>
#include <algorithm>
using namespace std;

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

// Main function to test the AVL Tree
int main() {
    AVLTree tree;
    double arr[] = {10.5, 20.3, 5.5, 6.7, 15.2, 10.5};
    int n = sizeof(arr) / sizeof(arr[0]);

    // Insert elements into the AVL tree
    for (int i = 0; i < n; i++) {
        tree.insert(arr[i]);
    }

    cout << "In-order traversal of the AVL tree: ";
    tree.display();

    // Search for a key
    double key = 15.2;
    cout << "Searching for " << key << ": " << (tree.search(key) ? "Found" : "Not Found") << endl;

    // Delete a key
    tree.deleteKey(10.5);
    cout << "In-order traversal after deleting ONE 10.5: ";
    tree.display();

    return 0;
}