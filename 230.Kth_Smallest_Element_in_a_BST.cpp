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
    int _idx = 1;
    int _ret = -1;

    int kthSmallest(TreeNode* root, int k) {
            
      
        if(root == NULL) return 0;
        bool result = traverse(root, k);
        return _ret;
        
        
    }
    bool traverse(TreeNode* root, int k){
        if(root == NULL) return false;
        bool left_ret = traverse(root->left, k);
        if(left_ret == true) return true;
        if(_idx == k){
            _ret = root -> val;
            return true;
        }
        _idx += 1;
        return traverse(root->right, k);
    }


};
