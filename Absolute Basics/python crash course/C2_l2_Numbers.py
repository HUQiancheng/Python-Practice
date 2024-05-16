# # when cursor changes its appearance simply use ENFG (Insert) to make it back to normal
# Integers Floats
m = ((1+3+1-4+2+1+2)/2) ** 3 - 0.1
print(m)

# # Avoiding Type Errors with the str() Function
# age = 23
# message = "Happy " + age + "rd Birthday!"
# print(message)

# ypeError: can only concatenate str (not "int") to str

# This is a type error. It means Python can’t recognize the kind of information
# you’re using. In this example Python sees at u that you’re using a variable
# that has an integer value (int), but it’s not sure how to interpret that
# value. Python knows that the variable could represent either the numerical
# value 23 or the characters 2 and 3. When you use integers within strings
# like this, you need to specify explicitly that you want Python to use the integer
# as a string of characters. You can do this by wrapping the variable in the
# str() function, which tells Python to represent non-string values as strings:

age = 23
print("I'm "+str(age)+" years old")

