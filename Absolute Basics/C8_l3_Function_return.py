# Return Values
# A function doesn’t always have to display its output directly. Instead, it can
# process some data and then return a value or set of values. The value the
# function returns is called a return value. The return statement takes a value
# from inside a function and sends it back to the line that called the function.
# Return values allow you to move much of your program’s grunt work into
# functions, which can simplify the body of your program.

# Returning a Simple Value
# def oper_three(num1, num2, num3=0):
#     """num3 has its default value when not given it shall be just zero"""
#     """This time return the value of num"""
#     num = num1 - num2 + (num1 + num2) * num3
#     return num
#
#
# A = [1, 5]
#
# p1 = oper_three(num2=A[1], num1=A[0], num3=1)
# p2 = oper_three(num1=A[1], num2=A[0])
#
# print(p1)
# print(p2)


# Making an Argument Optional

# # the incorrect example
#
# def mult_two(num1, num2):
#     if (num1 is None) and (num2 is not None):
#         fl = "num1 none"
#         return num2, fl
#     elif (num2 is not None) and (num1 is not None):
#         fl = "num2 none"
#         return num1, fl
#     elif (num1 is None) and (num2 is not None):
#         fl = "num1 and num2 none"
#         return 0, fl
#     else:
#         result = num1 * num2
#         fl = "result"
#         return result, fl
#

# p1, p2 = mult_two()
# print(p1, p2)
# p1, p2 = mult_two(num2=2)
# print(p1, p2)
# p1, p2 = mult_two(2, 3)
# print(p1, p2)

# The issue here is that the function mult_two() expects two arguments, num1 and num2,
# and you are not providing any default values for them. So, when you call the function
# without any arguments, Python raises an error because it expects both of them.

# To fix this issue, you can assign default values to the arguments in the function
# definition. In this case, you can set the default values to None:

# Same when with strings '' as it is equivalent as None for value in numbers

def mult_two(num1=None, num2=None):
    if (num1 is None) and (num2 is not None):
        fl = "num1 none"
        return num2, fl
    elif (num2 is None) and (num1 is not None):
        fl = "num2 none"
        return num1, fl
    elif (num1 is None) and (num2 is None):
        fl = "num1 and num2 none"
        return 0, fl
    else:
        result = num1 * num2
        fl = "result"
        return result, fl


# Now you can call the function without any arguments

p1, p2 = mult_two()
print(p1, p2)
p1, p2 = mult_two(num2=2)
print(p1, p2)
p1, p2 = mult_two(2, 3)
print(p1, p2)
