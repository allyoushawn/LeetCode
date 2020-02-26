/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        
        vector <int> ret;
        if(root == NULL){
            return ret;
        }
        else{
            ret.push_back(root->val);
        }
        
        if(root -> left != NULL){
            vector <int> left_vec = preorderTraversal(root->left);
            ret.insert(ret.end(), left_vec.begin(), left_vec.end());
        }
        if(root -> right != NULL){
            vector <int> right_vec = preorderTraversal(root->right);
            ret.insert(ret.end(), right_vec.begin(), right_vec.end());
        }
        return ret;
        
    }
};