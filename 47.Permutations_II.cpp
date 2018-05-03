class Solution {
public:
    
    void gen_unique_permute(int begin, vector<int>& nums, vector<vector<int>>& ret){
        if(begin == nums.size() - 1){
            ret.push_back(nums);
        }
        else{
            for(int i = begin; i < nums.size(); i++){
                if((nums[begin] == nums[i] && i != begin) || checkmiddle( nums,  i ,  begin) == false){
                    continue;
                }
                else{
                    swap(nums[begin], nums[i]);
                    gen_unique_permute(begin + 1, nums, ret);
                    swap(nums[begin], nums[i]); 
                }
            }
        }      
    }
    
    // Check whether there is an element between begin and i is the same as element-i
    // If true, it skips the swap, leave the swap to the previous same-valued one
    bool check_middle(vector<int>& nums, int i , int begin){
        for(int k = begin; k<i; k++)
            if(nums[i] == nums[k])
                return false;
        return true;
    }
    
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> ret;
        gen_unique_permute(0, nums, ret);
        
        return ret;
        
    }
};