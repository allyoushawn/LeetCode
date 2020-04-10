#!/usr/bin/env python3

def bisect_left(nums, n):
    lptr = 0
    rptr = len(nums) - 1
    if rptr < 0: return 0

    while rptr > lptr:
        mid = (lptr + rptr) // 2
        if nums[mid] >= n:
            rptr = mid
        else:
            lptr = mid + 1
    if n > nums[rptr]:
        return rptr + 1
    return rptr

l1 = [2, 2,2,2, 8, 10, 21, 13, 72]
n = 100
idx = bisect_left(l1, n)
print('{}, target: {}'.format(l1, n))
print(idx)
