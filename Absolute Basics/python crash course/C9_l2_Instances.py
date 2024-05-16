class Car():
    def __init__(self, make_var, model_var, year_var):
        self.make = make_var
        self.model = model_var
        self.year = year_var  # It is not that the attribute must consist the input var.
        # It could also be like self.year = year_var + 3, for 3 is the year when it is
        # actually in the auto shop. But if you are stupid enough, it could also be
        # self.year = year_var - make_var * model_var-( make_var + model_var)/3.
        # Though it does not have practical meaning, it is still theoratically correct
        # in class definition, for the attribute is firstly defined when you typed in this
        # self.year, not because there are only 3 input values here or the input value name seems
        # to include this year. You could add many possible input but the attribute is only in
        # sentence: self.attribute = ... defined

        # imported from gpt 4
        # You correctly pointed out that the attributes of an object do not necessarily have to
        # be exactly the same as the input parameters. The attributes can be set to any values
        # based on the input parameters or any other logic.
        # As long as the attribute assignment logic is valid and does not lead to errors, the
        # class definition will work. However, it is essential to keep the logic meaningful and
        # aligned with the class's intended purpose.

        self.odometer_reading = 0  # For example, here a default value is set to an attribute
        # now it only has 3 inpout but 4 attributes

    # The following methods and notation for attributes is eaxcatly the same in MATALB
    # See the end of the file...
    def get_discriptive_name(self):
        """Return a neatly formatted descriptive name."""
        long_name = str(self.year) + ' ' + self.make + ' ' + self.model
        return long_name.title()  # Since it is not a list but a string it needs to be returned

    def read_odometer(self):
        """Print a statement showing the car's mileage."""
        print("This car has " + str(self.odometer_reading) + " miles on it.")

    def update_odometer(self, mileage):
        """Set the odometer reading to the given value."""
        if mileage > self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("You can not roll back an odometer!!")

    def increment_odometer(self, miles):
        self.odometer_reading += miles


my_new_car = Car('audi', 'a4', 2016)
my_new_car.get_discriptive_name()

my_new_car.read_odometer()
my_new_car.odometer_reading = 23
my_new_car.read_odometer()
my_new_car.update_odometer(25)
my_new_car.read_odometer()
my_new_car.update_odometer(18)
my_new_car.read_odometer()
my_new_car.increment_odometer(13)
my_new_car.read_odometer()

# MATLAB Version
# % Define the Car class
# classdef Car
#     properties
#         make
#         model
#         year
#         odometer_reading = 0 % Set default value to an attribute
#     end
#
#     methods
#         % Constructor
#         function obj = Car(make_var, model_var, year_var)
#             obj.make = make_var;
#             obj.model = model_var;
#             obj.year = year_var;
#         end
#
#         % Get descriptive name
#         function name = get_discriptive_name(obj)
#             name = [num2str(obj.year), ' ', obj.make, ' ', obj.model];
#             % Since it is not a list but a string, it needs to be returned
#         end
#
#         % Read odometer
#         function read_odometer(obj)
#             fprintf('This car has %d miles on it.\n', obj.odometer_reading);
#         end
#
#         % Update odometer
#         function obj = update_odometer(obj, mileage)
#             if mileage > obj.odometer_reading
#                 obj.odometer_reading = mileage;
#             else
#                 disp('You can not roll back an odometer!');
#             end
#         end
#
#         % Increment odometer
#         function obj = increment_odometer(obj, miles)
#             obj.odometer_reading = obj.odometer_reading + miles;
#         end
#     end
# end
#
# % Test the Car class
# my_new_car = Car('audi', 'a4', 2016);
# disp(my_new_car.get_discriptive_name());
#
# my_new_car.read_odometer();
# my_new_car.odometer_reading = 23;
# my_new_car.read_odometer();
# my_new_car = my_new_car.update_odometer(25);
# my_new_car.read_odometer();
# my_new_car = my_new_car.update_odometer(18);
# my_new_car.read_odometer();
# my_new_car = my_new_car.increment_odometer(13);
# my_new_car.read_odometer();
