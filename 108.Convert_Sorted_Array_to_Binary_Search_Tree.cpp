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
    
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        if(nums.size() == 0) return NULL;
        return helper(nums, 0, nums.size() - 1);
        
    }
    TreeNode* helper(vector<int>& nums, int start_idx, int end_idx){ 
        if(end_idx == start_idx) {
            TreeNode* ret = new TreeNode(nums[start_idx]);
            return ret;
        }
        if(end_idx - start_idx == 1){
            TreeNode* ret =  new TreeNode(nums[end_idx]);
            ret-> left = new TreeNode(nums[start_idx]);
            return ret;
        }
        int mid_idx = start_idx + (end_idx - start_idx + 1) / 2;
        TreeNode*left_tree_root = helper(nums, start_idx, mid_idx - 1);
        TreeNode*right_tree_root = helper(nums, mid_idx + 1, end_idx);
        TreeNode* ret =  new TreeNode(nums[mid_idx]);
        ret->left = left_tree_root;
        ret-> right = right_tree_root;
        return ret;
    }
    
};
