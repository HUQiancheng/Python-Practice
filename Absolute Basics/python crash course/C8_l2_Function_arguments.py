# Passing Arguments
# Because a function definition can have multiple parameters, a function call
# may need multiple arguments. You can pass arguments to your functions
# in a number of ways. You can use positional arguments, which need to be in
# the same order the parameters were written; keyword arguments, where each
# argument consists of a variable name and a value; and lists and dictionaries
# of values. Let’s look at each of these in turn.

def add_two(num1, num2, num3=0):
    """num3 has its default value when not given it shall be just zero"""
    num = num1 - num2 + (num1+num2)*num3
    print(num)


A = []
for i in range(0, 2):
    a = input("Type a number:\n")
    A.append(int(a))

add_two(A[0], A[1], 3)
add_two(A[1], A[0], 3)
add_two(A[1], 9)

add_two(num1=A[0], num2=A[1], num3=1)
add_two(num2=A[1], num1=A[0], num3=1)
add_two(num1=A[1], num2=A[1])

