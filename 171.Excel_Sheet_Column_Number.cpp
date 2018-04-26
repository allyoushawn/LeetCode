class Solution {
public:
    int titleToNumber(string s) {
        int num = 0;
        for(int i = s.size() - 1; i >=0; i--){
            num += pow(26, s.size() - 1 - i) * (s[i] - 'A' + 1);
        }
        return num;
        
    }
};
