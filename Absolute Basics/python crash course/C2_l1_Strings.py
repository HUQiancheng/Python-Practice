# 注释已经改成了 ctrl+#

# A string is simply a series of characters. Anything inside quotes is considered
# a string in Python, and you can use single or double quotes around
# your strings like this:
msg = "This is a string."
print(msg)
msg = 'This is also a string'
print(msg)

# A string is simply a series of characters. Anything inside quotes is considered
# a string in Python, and you can use single or double quotes around
# your strings like this:

msg = 'I told my friend, "Python is my favorite language!"'
print(msg)
msg = "The language 'Python' is named after Monty Python, not the snake."
print(msg)
msg = "One of Python's strengths is its diverse and supportive community."
print(msg)

# Changing Case in a String with Methods
print(msg.upper())
print(msg.lower())
print(msg.title())

msg='Python is great    !     '
print(msg.rstrip())

# Combining or Concatenating Strings
str0 = "I'm"
str1 = "Qiancheng Hu"
str2 = "Yikun Hao"
str3 = "Pengfei Yang"
print('\n\n' + str0 + " " + str1 + "\n\tnot    ".rstrip() + ' ' + str2 + '\n\tor ' + str3)
# so the ".rstrip" is a method to get rid of all the blanks after a string
# That means it could strip the white space
# Also in this case we can see that the symbol '\n' and '\t' is
# is just like typing Tab and Enter
print(str3[::2])
print(str3[::1])
print(str3[::])
print(str3[::-1])
print(str3[::-2])
