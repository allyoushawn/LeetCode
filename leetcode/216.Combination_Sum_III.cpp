class Solution {
public:
   vector<vector<int>> recursive(int k, int n, int begin){
        vector<vector<int>> ret;
        if(k == 1 && n <= 9 && n == begin){
            vector<int> v = {n};
            ret.push_back(v);
        }
        else{
            for(int i = begin + 1; i < 10; i++){
                if(n - i <= 0) break;
                vector<vector<int>> partial_ret = recursive(k - 1, n - begin, i);
                for(int idx = 0 ; idx < partial_ret.size(); idx ++){
                    partial_ret[idx].push_back(begin);
                    ret.push_back(partial_ret[idx]);    
                }
            }   
        }
       return ret;
       
   }
    
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> ret;
        
        if(n > 45 || n <= 0 || k > 9) return ret;
        for(int i = 1; i <= n; i++){
            vector<vector<int>> partial_ret = recursive(k, n, i);
            for(int idx = 0 ; idx < partial_ret.size(); idx ++){
                ret.push_back(partial_ret[idx]);
            }
        }
        return ret;
        
    }
};

/* v2
class Solution {
public:
   
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> ret;
        
        if(n > 45 || n <= 0 || k > 9) return ret;
        if(k == 1 && n < 9){
            vector<int> v = {n};
            ret.push_back(v);
        }
        else{
            for(int i = 1; i < 10; i++){
                if(n - i <= 0) break;
                partial_ret = combinationSum3(k - 1, n -i);
                for(int idx = 0 ; idx < partial_ret.size(); idx ++){
                    partial_ret[idx].push_back(i);
                    ret.push_back(partial_ret[idx]);    
            }   
        }     
        return ret;
        
    }
};*/





/* v1
class Solution {
public:
    void recursive(int leaved_num, int sum, int begin, vector<vector<int>>& ret){
        if(leaved_num == 1){
            if(sum == begin){
                vector<int> v = {begin};
                ret.push_back(v);
                return;
            }
        }
        else if(begin >= sum)  return;
        else{
            for(int j = begin + 1; j < 10; j++){
                recursive(leaved_num - 1, sum - begin, j, ret);
                for(int idx = 0 ; idx < ret.size(); idx ++){
                    ret[idx].push_back(begin);
                }
            }
        }

    }
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> ret;
        
        if(n > 45 || n <= 0 || k > 9) return ret;
        for(int i = 1; i < 10; i++){
            recursive(k, n, i, ret);
            
        }        
        return ret;
        
    }
};*/