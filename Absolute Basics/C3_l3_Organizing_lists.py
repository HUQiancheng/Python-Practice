# Often, your lists will be created in an unpredictable order, because you can’t
# always control the order in which your users provide their data. Although
# this is unavoidable in most circumstances, you’ll frequently want to present
# your information in a particular order. Sometimes you’ll want to preserve the
# original order of your list, and other times you’ll want to change the original
# order. Python provides a number of different ways to organize your lists,
# depending on the situation.

# Sorting a List Permanently with the sort() Method
auto = ['door', 'wheel', 'steer', 'light', 'engine', 'seat']
auto.sort()
print(auto)  # so it is not reversible, we cannot put it back to the original order
# but it does not mean that we cannot modify it, so it is not permanent
auto.insert(2, "rearview mirror")
auto.sort()
print(auto)
# or we can do it in a reverse way
auto.sort(reverse=True)
print(auto)

# Sorting a List Temporarily with the sorted() Function
comp = sorted(auto)
print("\n\nSorting a List Temporarily with the sorted() Function \n\nOriginal\t")
print(auto)
print("\nSorted\t")
print(comp)

# Printing a List in Reverse Order
auto = ['door', 'wheel', 'steer', 'light', 'engine', 'seat']
print("\n\n")
print(auto)
auto.reverse()
print("\nNotice that reverse() doesn’t sort backward alphabetically; it simply \nreverses the order of the list:")
print(auto)

# # Error example
# motorcycles = []
# print(motorcycles[-1])  # There is no element inside the list so -1 here is meaningless
# # IndexError: list index out of range
