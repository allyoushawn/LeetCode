class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> ret(nums.size(), 1);
        for(int i = 1; i < nums.size(); i++){
            ret[i] *= ret[i - 1] * nums[i - 1];
        }
        int prod = 1;
        for(int i = nums.size() - 2; i >=0; i--){
            prod *= nums[i + 1];
            ret[i] *= prod;
        }
        
    return ret;
    }
};
