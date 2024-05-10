# Defining a Function
def hello_my_friend():
    """The triple ' here is just a function docstring to explain what it does"""
    """so it is a function to greet users"""
    print("Hi, my friend. Welcome back to Python learning!")


hello_my_friend()


# Passing Information to a Function
def hello(name):
    """adding name"""
    print("How are you my friend " + name.title() + "?")


hello("hu")

