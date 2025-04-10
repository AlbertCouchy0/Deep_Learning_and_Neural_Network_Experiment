def find_smallest_two(lst):
    if len(lst) < 2:
        return lst, list(range(len(lst)))
    min1, min2 = (lst[0], 0), (lst[1], 1)
    if min1[0] > min2[0]:
        min1, min2 = min2, min1
    for i in range(2, len(lst)):
        if lst[i] < min1[0]:
            min2 = min1
            min1 = (lst[i], i)
        elif lst[i] < min2[0]:
            min2 = (lst[i], i)
    return [min1[0], min2[0]], [min1[1], min2[1]]

my_list = [10, 5, 20, 8, 15]
smallest_two_values, smallest_two_indices = find_smallest_two(my_list)
print("最小的两个元素:", smallest_two_values)
print("它们对应的序号:", smallest_two_indices)
