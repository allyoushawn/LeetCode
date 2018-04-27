class Solution {
public:
    bool isAnagram(string s, string t) {
    /*
        // Beats 16%
        int char_num = 26;
        int count[char_num] = {0};
        if(s.length() != t.length())  return false;
        for(int i = 0; i <s.length(); i ++){
            count[s[i] - 'a'] -= 1;
            count[t[i] - 'a'] += 1;
            
        }
        for(int i = 0 ; i < char_num; i++){
            if(count[i]!= 0)  return false;
        }
        return true;
    }
    */
        // Beats 99.39 %, returning judgment as early as possible makes huge difference
        int char_num = 26;
        int count[char_num] = {0};
        if(s.length() != t.length())  return false;
        for(int i = 0; i <s.length(); i ++){
            count[s[i] - 'a'] += 1;
            
        }
        for(int i = 0 ; i < t.length(); i++){
            count[t[i] - 'a'] -= 1;
            if(count[t[i] - 'a'] < 0) return false; 
        }
        return true;
};