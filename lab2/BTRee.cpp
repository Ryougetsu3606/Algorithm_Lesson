// This file implements an AVL Tree, a self-balancing binary search tree.
// The AVL Tree ensures that the height difference (balance factor) between
// the left and right subtrees of any node is at most 1, maintaining O(log n)
// time complexity for insertion, deletion, and search operations.

#include <iostream>
#include <algorithm>
using namespace std;

class AVLTree {
private:
    // Node structure representing a single node in the AVL Tree.
    struct Node {
        double key;       // The key value stored in the node.
        Node* left;       // Pointer to the left child.
        Node* right;      // Pointer to the right child.
        int height;       // Height of the node in the tree.
        Node(double k) : key(k), left(nullptr), right(nullptr), height(1) {}
    };

    Node* root; // Pointer to the root node of the AVL Tree.

    // Helper function to get the height of a node.
    int getHeight(Node* node) {
        return node ? node->height : 0;
    }

    // Helper function to calculate the balance factor of a node.
    int getBalanceFactor(Node* node) {
        return node ? getHeight(node->left) - getHeight(node->right) : 0;
    }

    // Performs a right rotation on the given node.
    Node* rotateRight(Node* y) {
        Node* x = y->left;
        Node* T2 = x->right;

        // Perform rotation.
        x->right = y;
        y->left = T2;

        // Update heights.
        y->height = max(getHeight(y->left), getHeight(y->right)) + 1;
        x->height = max(getHeight(x->left), getHeight(x->right)) + 1;

        return x; // Return the new root after rotation.
    }

    // Performs a left rotation on the given node.
    Node* rotateLeft(Node* x) {
        Node* y = x->right;
        Node* T2 = y->left;

        // Perform rotation.
        y->left = x;
        x->right = T2;

        // Update heights.
        x->height = max(getHeight(x->left), getHeight(x->right)) + 1;
        y->height = max(getHeight(y->left), getHeight(y->right)) + 1;

        return y; // Return the new root after rotation.
    }

    // Recursive function to insert a key into the AVL Tree.
    Node* insert(Node* node, double key) {
        if (!node) return new Node(key); // Create a new node if the tree is empty.

        // Traverse the tree to find the correct position for the new key.
        if (key < node->key)
            node->left = insert(node->left, key);
        else if (key > node->key)
            node->right = insert(node->right, key);
        else
            return node; // Duplicate keys are not allowed.

        // Update the height of the current node.
        node->height = 1 + max(getHeight(node->left), getHeight(node->right));

        // Get the balance factor to check if the node is unbalanced.
        int balance = getBalanceFactor(node);

        // Perform rotations to balance the tree.
        if (balance > 1 && key < node->left->key)
            return rotateRight(node);
        if (balance < -1 && key > node->right->key)
            return rotateLeft(node);
        if (balance > 1 && key > node->left->key) {
            node->left = rotateLeft(node->left);
            return rotateRight(node);
        }
        if (balance < -1 && key < node->right->key) {
            node->right = rotateRight(node->right);
            return rotateLeft(node);
        }

        return node; // Return the unchanged node pointer.
    }

    // Helper function to find the node with the smallest key in a subtree.
    Node* minValueNode(Node* node) {
        Node* current = node;
        while (current->left)
            current = current->left;
        return current;
    }

    // Recursive function to delete a key from the AVL Tree.
    Node* deleteNode(Node* root, double key) {
        if (!root) return root; // Base case: the tree is empty.

        // Traverse the tree to find the node to be deleted.
        if (key < root->key)
            root->left = deleteNode(root->left, key);
        else if (key > root->key)
            root->right = deleteNode(root->right, key);
        else {
            // Node with only one child or no child.
            if (!root->left || !root->right) {
                Node* temp = root->left ? root->left : root->right;
                if (!temp) {
                    temp = root;
                    root = nullptr;
                } else
                    *root = *temp; // Copy the contents of the non-empty child.
                delete temp;
            } else {
                // Node with two children: Get the inorder successor.
                Node* temp = minValueNode(root->right);
                root->key = temp->key; // Copy the inorder successor's key.
                root->right = deleteNode(root->right, temp->key); // Delete the successor.
            }
        }

        if (!root) return root; // If the tree had only one node.

        // Update the height of the current node.
        root->height = 1 + max(getHeight(root->left), getHeight(root->right));

        // Get the balance factor to check if the node is unbalanced.
        int balance = getBalanceFactor(root);

        // Perform rotations to balance the tree.
        if (balance > 1 && getBalanceFactor(root->left) >= 0)
            return rotateRight(root);
        if (balance > 1 && getBalanceFactor(root->left) < 0) {
            root->left = rotateLeft(root->left);
            return rotateRight(root);
        }
        if (balance < -1 && getBalanceFactor(root->right) <= 0)
            return rotateLeft(root);
        if (balance < -1 && getBalanceFactor(root->right) > 0) {
            root->right = rotateRight(root->right);
            return rotateLeft(root);
        }

        return root; // Return the updated root.
    }

    // Recursive function to search for a key in the AVL Tree.
    bool search(Node* node, double key) {
        if (!node) return false; // Base case: the tree is empty.
        if (node->key == key) return true; // Key found.
        if (key < node->key)
            return search(node->left, key); // Search in the left subtree.
        else
            return search(node->right, key); // Search in the right subtree.
    }

    // Recursive function to perform an inorder traversal of the AVL Tree.
    void inorder(Node* node) {
        if (node) {
            inorder(node->left); // Visit the left subtree.
            cout << node->key << " "; // Print the key.
            inorder(node->right); // Visit the right subtree.
        }
    }

public:
    // Constructor to initialize an empty AVL Tree.
    AVLTree() : root(nullptr) {}

    // Public function to insert a key into the AVL Tree.
    void insert(double key) {
        root = insert(root, key);
    }

    // Public function to delete a key from the AVL Tree.
    void deleteKey(double key) {
        root = deleteNode(root, key);
    }

    // Public function to search for a key in the AVL Tree.
    bool search(double key) {
        return search(root, key);
    }

    // Public function to display the AVL Tree using inorder traversal.
    void display() {
        inorder(root);
        cout << endl;
    }
};

int main() {
    AVLTree tree;
    double arr[] = {10, 20, 30, 40, 50, 25};
    int n = sizeof(arr) / sizeof(arr[0]);

    // Insert elements into the AVL Tree.
    for (int i = 0; i < n; i++) {
        tree.insert(arr[i]);
    }

    // Display the AVL Tree.
    cout << "Inorder traversal of the AVL tree: ";
    tree.display();

    // Delete a key from the AVL Tree.
    tree.deleteKey(30);
    cout << "After deleting 30: ";
    tree.display();

    // Search for a key in the AVL Tree.
    cout << "Search 50: " << (tree.search(50) ? "Found" : "Not Found") << endl;

    return 0;
}
