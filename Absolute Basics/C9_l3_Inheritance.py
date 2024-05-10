class Car():
    """A simple attempt to represent a car."""

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0
        self.tank = "Full"

    def get_descriptive_name(self):
        long_name = str(self.year) + ' ' + self.make + ' ' + self.model
        return long_name.title()

    def read_odometer(self):
        print("This car has " + str(self.odometer_reading) + " miles on it.")

    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("You can't roll back an odometer!")

    def increment_odometer(self, miles):
        self.odometer_reading += miles

    def fill_gas_tank(self):
        print(self.tank)


class Battery():
    """A simple attempt to model a battery for an electric car."""

    def __init__(self, battery_size=70):
        """Initialize the battery's attributes."""
        self.battery_size = battery_size

    def describe_battery(self):
        """Print a statement describing the battery size."""
        print("This car has a " + str(self.battery_size) + "-kWh battery.")


class ElectricCar(Car):  # When you create a child class, the parent class
    # must be part of the current file and must appear before the child class in
    # the file
    """Represent aspects of a car, specific to electric vehicles."""

    def __init__(self, make, model, year):
        """Initialize attributes of the parent class."""

        super().__init__(make, model, year)  # call the __init__() method from ElectricCar’s parent class
        self.battery_size = Battery()  # Instances as Attributes

    # # original code instead of instances as Attributes for battery_size
    #     self.battery_size = 70  # Defining Attributes
    #
    # def describe_battery(self):  # Defining Methods
    #     """Print a statement describing the battery size."""
    #     print("This car has a " + str(self.battery_size) + "-kWh battery.")

    def fill_gas_tank(self):  # Overriding
        """Electric cars don't have gas tanks."""
        print("This car doesn't need a gas tank!")


my_tesla = ElectricCar('tesla', 'model s', 2016)
print(my_tesla.get_descriptive_name())
