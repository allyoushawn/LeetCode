class Solution {
public:
    
    void gen_permute(int begin, vector<int>& nums, vector<vector<int>>& ret){
        if(begin == nums.size() - 1){
            ret.push_back(nums);
        }
        else{
            for(int i = begin; i < nums.size(); i++){
                swap(nums[begin], nums[i]);
                gen_permute(begin + 1, nums, ret);
                swap(nums[begin], nums[i]);          
            }
        }      
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ret;
        gen_permute(0, nums, ret);
        
        return ret;
        
    }
};