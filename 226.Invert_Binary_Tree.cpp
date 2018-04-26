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
    TreeNode* invertTree(TreeNode* root) {
        
        
        if(root == NULL || (root->left == NULL && root->right == NULL))  return root;
        
        TreeNode *tmp_node;
        if(root->left == NULL){
            root->left = invertTree(root->right);
            root->right = NULL;
        }
        else if(root->right == NULL){
            root->right = invertTree(root->left);
            root->left = NULL;
        }
        else{
            tmp_node = root->left;
            root->left = invertTree(root->right);
            root->right = invertTree(tmp_node);
        }
        return root;

            
        
    }
};
