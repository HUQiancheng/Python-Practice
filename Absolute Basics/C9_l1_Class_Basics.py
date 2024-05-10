class Dog():
    """A simple attempt to model a dog."""
    # def _init_(self, name, age):
    #     """Initialize name and age attributes."""
    #     self.name = name
    #     self.age = age
    # The problem in this instance is the incorrect
    # usage of the special method __init__. You have u
    # sed single underscores on both sides of the "init"
    # instead of double underscores.
    def __init__(self, name, age):
        """Initialize name and age attributes."""
        self.name = name
        self.age = age

    def sit(self):
        """Simulate a dog sitting in response to a command."""
        print(self.name.title() + " is now sitting.")

    def roll_over(self):
        """Simulate rolling over in response to a command."""
        print(self.name.title() + " rolled over!")


my_dog = Dog('allen', 2)
his_dog = Dog('billy', 3)
her_dog = Dog('klaus', 1)

print(my_dog.name.title() + " " + his_dog.name.title())
my_dog.roll_over()












