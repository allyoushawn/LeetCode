#!/usr/bin/env pyhton3

def binary_search(nums, n):
    start = 0
    end = len(nums) - 1
    while end - start > 1:
        mid = (end + start) // 2
        if nums[mid] < n:
            start = mid
        elif nums[mid] > n:
            end = start
        else:
            return mid
    return start

l1 = [2, 8, 10, 21, 13, 72]
n = 15
idx = binary_search(l1, n)
l1 = l1[:idx+1] + [n] + l1[idx+1:]
print(l1)
