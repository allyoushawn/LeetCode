class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int index1 = 0;
        int index2 = numbers.size() - 1;
        while(true){
            if(numbers[index1] + numbers[index2] < target) index1 += 1;
            
            else if(numbers[index1] + numbers[index2] > target) index2 -=1;
            else{
                vector<int> ret;
                // Convert from zero-based index to 1-based index
                ret.push_back(index1 + 1);
                ret.push_back(index2 + 1);
                return ret;
            }

                       
            
        }
        
    }
};