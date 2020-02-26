class Solution {
public:
    vector<int> diffWaysToCompute(string input) {
        vector<int> ret;
        int cur_idx = 0;
        while(cur_idx < input.length()){
            
            if(input[cur_idx] != '*' && input[cur_idx] != '-' && input[cur_idx] != '+'){
                cur_idx += 1;
                continue;
            }
                
        
            vector<int> part1 = diffWaysToCompute(input.substr(0, cur_idx));
            vector<int> part2 = diffWaysToCompute(input.substr(cur_idx + 1));

            for(int i = 0; i < part1.size(); i++){
                for(int j = 0; j < part2.size(); j++){
                    if(input[cur_idx] == '*' )
                        ret.push_back(part1[i] * part2[j]);
                    else if(input[cur_idx] == '+' )
                        ret.push_back(part1[i] + part2[j]);
                    else if(input[cur_idx] == '-')
                        ret.push_back(part1[i] - part2[j]);
                }
            
            }
            cur_idx += 1;
        }
        
        if(ret.size() == 0) ret.push_back(atoi(input.c_str()));
        
        return ret;
        
            
        
    }
};
