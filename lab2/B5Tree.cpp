#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

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

// Main function for testing
int main() {
    BTree tree;
    vector<double> data = {1.9,4.2,4.1,5.4,1.8,3.7,9.8,1.5,3.5,3.7,4.4,2.3,0.0,7.5,8.1,6.7,3.1,1.3,6.3,0.1,3.9,3.2,8.0,7.6,8.9,6.0,3.1,7.7,4.2,7.7,9.8,6.9,0.3,7.0,6.6,1.5,5.7,9.2,1.2,9.0,6.7,8.5,3.1,6.6,5.2,6.1,0.3,6.6,2.8,9.7,6.8,8.3,0.4,1.7,2.1,1.8,5.0,0.5,6.8,4.3,6.4,4.6,4.3,2.1,9.6,7.9,5.8,9.1,5.2,7.4,3.4,2.1,5.7,2.4,6.3,9.3,6.6,4.6,0.5,4.8,6.5,2.2,7.2,9.1,2.5,2.7,6.2,2.0,7.2,0.6,8.6,8.4,1.2,7.2,3.4,4.8,7.5,8.9,2.4,2.1};

    for (double key : data) {
        tree.insert(key);
    }

    cout << "B-Tree after insertion:" << endl;
    tree.print();

    tree.remove(7.3);
    cout << "\nB-Tree after removing 7.3:" << endl;
    tree.print();

    tree.remove(13);
    cout << "\nB-Tree after removing 13 (non-existent):" << endl;
    tree.print();

    tree.remove(0.8);
    cout << "\nB-Tree after removing 7:" << endl;
    tree.print();

    return 0;
}