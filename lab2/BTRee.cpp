#include <iostream>
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

// Main function to test the BinarySearchTree class
int main() {
    BinarySearchTree bst;

    // Insert elements into the tree
    bst.insert(5.5);
    bst.insert(3.3);
    bst.insert(7.7);
    bst.insert(3.3); // Duplicate insertion
    bst.insert(6.6);

    // Print the tree in-order
    cout << "In-order traversal: ";
    bst.printInOrder();

    // Search for elements
    cout << "Search 3.3: " << (bst.search(3.3) ? "Found" : "Not Found") << endl;
    cout << "Search 8.8: " << (bst.search(8.8) ? "Found" : "Not Found") << endl;

    // Delete an element
    bst.remove(3.3);
    cout << "After deleting 3.3, in-order traversal: ";
    bst.printInOrder();

    return 0;
}