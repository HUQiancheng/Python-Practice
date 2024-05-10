# Lists are mutable objects which could be modified anywhere
# -> see File: Q1_immutable_mutable_objects.py

# When you pass a list to a function, the function can modify the list. Any
# changes made to the list inside the function’s body are permanent, allowing
# you to work efficiently even when you’re dealing with large amounts of data.

# List modification
# a list can be modified inside a function without returning it. This is because
# lists are mutable objects, and when you pass a list to a function, you are
# actually passing a reference to the list object in memory. So, any changes
# you make to the list inside the function will be reflected in the original
# list outside the function.
#
# Here's an example to illustrate this:
def modify_list(lst):
    lst.append("new_element")


my_list = [1, 2, 3]
print("Before modification:", my_list)

modify_list(my_list)
print("After modification:", my_list)


# Output:
# Before modification: [1, 2, 3]
# After modification: [1, 2, 3, 'new_element']
# As you can see, even though the modify_list function does not return the modified
# list, the original list my_list is still updated with the new element. This is
# because the function works with a reference to the same list object in memory.

# List prevent modification

# Example 1
def process_list_while_loop(numbers):
    i = 0
    while i < len(numbers):
        if numbers[i] % 2 == 0:
            del numbers[i]
        else:
            i += 1
    return numbers


def process_list_for_loop(numbers):
    new_list = [num for num in numbers if num % 2 != 0]
    return new_list


original_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
processed_list_while = process_list_while_loop(original_list.copy())
processed_list_for = process_list_for_loop(original_list.copy())

print("Original list:", original_list)
print("Processed list using while loop:", processed_list_while)
print("Processed list using for loop:", processed_list_for)

# In this example, the while loop is used to remove even numbers from
# the list in-place, while the for loop creates a new list containing
# only odd numbers. In this specific case, the while loop is better
# suited for modifying the list in-place, as it has better control over
# the index variable.


#
def find_first_negative_while_loop(numbers):
    i = 0
    found = None
    while i < len(numbers) and found is None:
        if numbers[i] < 0:
            found = i
        i += 1
    return found


def find_first_negative_for_loop(numbers):
    found = None
    for i, num in enumerate(numbers):
        if num < 0:
            found = i
            break
    return found


numbers_list = [1, 2, 3, -1, 4, -2, 5]
first_negative_while = find_first_negative_while_loop(numbers_list)
first_negative_for = find_first_negative_for_loop(numbers_list)

print("First negative index using while loop:", first_negative_while)
print("First negative index using for loop:", first_negative_for)

# In this example, we use a while loop and a for loop to find the
# index of the first negative number in a list. The while loop
# has a specific termination condition that stops iterating when
# the first negative number is found. In the for loop, we use the
# 'break' statement to achieve the same effect. While both loops can
# achieve the same result, the while loop allows for more fine-grained
# control over the loop condition, whereas the for loop requires a 'break'
# statement to terminate early.
