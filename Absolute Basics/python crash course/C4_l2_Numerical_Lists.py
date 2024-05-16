# Using the range() Function
for val in range(1, 5):
    print(val)
    # The range() function causes Python to start counting at the first
    # value you give it, and it stops when it reaches the second value you provide.
    # Because it stops at that second value, the output never contains the end
    # value, which would have been 5 in this case.

# Using range() to Make a List of Numbers
numl = list(range(1, 6))
print(numl)
enuml = list(range(0, 12, 2))
# range(#starts,#ends,#interval)
# range never reaches its end
print(enuml)
print("\nForm I")
squares = []
for val in enuml:
    s = val ** 2
    squares.append(s)
print(squares)
# or it can be like this by predefining a fixed length of a list and then
# specify value to each one of the element inside of it

print("\nZero Matrix")
leng = len(enuml)
squares = [0] * leng
print(squares)
for i in range(0, leng):
    squares[i] = enuml[i] ** 2
print("\nForm II")
print(squares)

# use Rename to modify the variables Shift+F6, Enter to confirm

# Simple Statistics with a List of Numbers
print("\nSimple Statistics:")
digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
print(min(digits))
print(max(digits))
print(sum(digits))

# List Comprehensions
print("List Comprehensions")
squares = [value ** 2 for value in enuml]
print(squares)
