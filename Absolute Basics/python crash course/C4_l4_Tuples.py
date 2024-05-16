# Lists work well for storing sets of items that can change throughout the
# life of a program. The ability to modify lists is particularly important when
# you’re working with a list of users on a website or a list of characters in a
# game. However, sometimes you’ll want to create a list of items that cannot
# change. Tuples allow you to do just that. Python refers to values that cannot
# change as immutable, and an immutable list is called a tuple.


# Defining a Tuple
print("\nForm I")
dimensions = (1, 2, 45, 3, -4, "what??")
for i in range(1, len(dimensions)):
    print(dimensions[i])
print("\nForm II")
# or in this form
for i in dimensions[1:]:
    print(i)

# Let’s see what happens if we try to change one of the items in the tuple
# dimensions:

# dimensions[0] = 250
# Tuples don't support item assignment TypeError: 'tuple' object does not support item assignment

# Looping Through All Values in a Tuple
# You can loop over all the values in a tuple using a for loop, just as you did
# with a list:

# Writing over a Tuple

# Python doesn’t raise any errors this time, because
# overwriting a variable is always valid:
print("\nOverwriting:")
dimensions = (10, 4, -3, -14, "Yeah!!")
for i in dimensions[1:]:
    print(i)

# Page 75
