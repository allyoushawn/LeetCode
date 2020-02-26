class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        int xor_result = 0;
        for(int i = 0; i < nums.size(); i++){
            xor_result ^= nums[i];
        }
        
        int diff = xor_result & -xor_result;
        int ret1 = 0, ret2 = 0;
        for(int i = 0; i < nums.size(); i++){
            
            if(nums[i] & diff)
                ret1 ^= nums[i];
            
            else
                ret2 ^= nums[i];
        }
        
        vector<int> ret { ret1 ,ret2 };
        return ret;
        
    }
};
