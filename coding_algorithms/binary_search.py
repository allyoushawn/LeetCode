#!/usr/bin/env pyhton3

def binary_search(nums, n):
    start = 0
    end = len(nums) - 1
    while end >= start:
        mid = start + (end - start) // 2
        if nums[mid] == n: return mid

        if nums[mid] < n:
            start = mid + 1
        elif nums[mid] > n:
            end = mid - 1
    return -1

l1 = [2, 8, 10, 21, 13, 72]
n = 15
idx = binary_search(l1, n)
print('{}, target: {}'.format(l1, n))
print(idx)
