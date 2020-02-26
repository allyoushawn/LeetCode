class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int move_forward_count = 0;
        for(int i = 0 ; i < nums.size(); i++){
            if(nums[i] == 0) 
                move_forward_count += 1;
            else{
                nums[i - move_forward_count] = nums[i];
            }
        }
        for(int i = nums.size() - 1; i > nums.size() - 1 - move_forward_count; i--){
            nums[i] = 0;
            
        }
        
    }
};
