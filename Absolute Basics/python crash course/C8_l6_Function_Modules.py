# One advantage of functions is the way they separate blocks of code from
# your main program. By using descriptive names for your functions, your
# main program will be much easier to follow. You can go a step further by
# storing your functions in a separate file called a module and then importing
# that module into your main program. An import statement tells Python to
# make the code in a module available in the currently running program file.
# Storing your functions in a separate file allows you to hide the details of
# your program’s code and focus on its higher-level logic. It also allows you to
# reuse functions in many different programs. When you store your functions
# in separate files, you can share those files with other programmers without
# having to share your entire program. Knowing how to import functions
# also allows you to use libraries of functions that other programmers have
# written.
# There are several ways to import a module, and I’ll show you each of
# these briefly

# they are just like m files in MATLAB as a script

# Importing an Entire Module
import pizza  # pizza is module
pizza.make_pizza(16, 'pepperoni')  # make_pizza is the function defined in pizza
pizza.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')

# Importing Specific Functions
# from module_name import function_name

# Using as to Give a Function an Alias
# from module_name import function_name as fn

# Using as to Give a Module an Alias
# import module_name as mn

# Importing All Functions in a Module
# from module_name import *
