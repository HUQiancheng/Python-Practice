# Passing an Arbitrary Number of Arguments
def make_pizza(*toppings):
    """Print the list of toppings that have been requested."""
    print(toppings)


make_pizza('pepperoni')
make_pizza('mushrooms', 'green peppers', 'extra cheese')


# Mixing Positional and Arbitrary Arguments

def make_pizza(size, *toppings):
    """Summarize the pizza we are about to make."""
    print("\nMaking a " + str(size) + "-inch pizza with the following toppings:")

    for topping in toppings:
        print("- " + topping)


make_pizza(16, 'pepperoni')
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')


# Using Arbitrary Keyword Arguments
def build_profile(first, last, **user_info):
    """Build a dictionary containing everything we know about a user."""
    profile = {'first_name': first, 'last_name': last}
    for key, value in user_info.items():
        profile[key] = value
    return profile


user_profile = build_profile('albert', 'einstein', location='princeton', field='physics')
print(user_profile)

# The ** before user_info in the function definition is used to indicate that user_info
# should accept any number of keyword arguments. These keyword arguments are then passed
# to the function as a dictionary.
#
# In the given example, user_info is a dictionary that collects any key-value pairs passed
# to the function when it is called, beyond the required first and last arguments. This allows
# you to pass any number of additional keyword arguments to the function, which will then be
# included in the profile dictionary.
#
# Here's another example using **kwargs to demonstrate how it can be used:


def print_info(name, age, **extra_info):
    print(f"Name: {name}")
    print(f"Age: {age}")

    for key, value in extra_info.items():
        print(f"{key.capitalize()}: {value}")


print_info("Alice", 30, country="USA", occupation="Software Engineer", hobby="Hiking")
# In this example, the print_info function takes a name and age argument,
# as well as any number of additional keyword arguments.
