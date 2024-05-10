# # the lesson will be mastered when necessary
# my_dict = {'name': 'Andrei Neagoie', 'age': 30, 'magic_power': False}
# print(my_dict['name'])  # Andrei Neagoie
# print(len(my_dict))  # 3
# print(list(my_dict.keys()))  # ['name', 'age', 'magic_power']
# print(list(my_dict.values()))  # ['Andrei Neagoie', 30, False]
# print(list(my_dict.items()))  # [('name', 'Andrei Neagoie'), ('age', 30),('magic_power', False)]
# my_dict['favourite_snack'] = 'Grapes'  # {'name': 'Andrei Neagoie', 'age': 30,'magic_power': False, 'favourite_snack':
# # 'Grapes'}
# print(my_dict.get('age'))  # 30 --> Returns None if key does not exist.
# print(my_dict.get('ages', 0))  # 0 --> Returns default (2nd param) if key is not found
#
# # Remove key
# del my_dict['name']
# my_dict.pop('name', None)
#
# print(list(my_dict.items()))
#
# my_dict.update({'cool': True})
# # {'name':
# # 'Andrei Neagoie', 'age': 30, 'magic_power': False, 'favourite_snack': 'Grapes',
# # 'cool': True}
# print(list(my_dict.items()))
# # Unsure
#
# # {**my_dict, **{'cool': False}}  # {'name':
# # # 'Andrei Neagoie', 'age': 30, 'magic_power': False, 'favourite_snack': 'Grapes',
# # # 'cool': False}
# # print(list(my_dict.items()))
# # new_dict = dict([['name', 'Andrei'], ['age', 32], ['magic_power', False]])  # Creates
# # # a dict from collection of key-value pairs.
# # new_dict = dict(zip(['name', 'age', 'magic_power'], ['Andrei', 32, False]))  # Creates
# # # a dict from two collections.
# # new_dict = my_dict.pop('favourite_snack')  # Removes
# # # item from dictionary.

# Create an empty dictionary using {}.
employees = {}
print(employees)  # Output: {}
print(type(employees))  # Output: <class 'dict'>

# Create a dictionary with items using {}.
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(employees)  # Output: {1: 'Tom', 2: 'Macy', 3: 'Sam'}

# Create an empty dictionary using dict() constructor.
employees = dict()
print(employees)  # Output: {}
print(type(employees))  # Output: <class 'dict'>

# Create a dictionary using dict() with a mapping(in this case, another dict) as input.
employees = dict({1: 'Tom', 2: 'Macy', 3: 'Sam'})
print(employees)  # Output: {1: 'Tom', 2: 'Macy', 3: 'Sam'}

# # Create a dictionary using dict() with an iterable as input.
# Input is a list of tuples, each of which becomes a key-value pair.
employees = dict([(1, 'Tom'), (2, 'Macy'), (3, 'Sam')])
print(employees)  # Output: {1: 'Tom', 2: 'Macy', 3: 'Sam'}
# Input is an iterable created using zip() method that is taking lists of keys and values.
employees = dict(zip([1, 2, 3], ['Tom', 'Macy', 'Sam']))
print(employees)  # Output: {1: 'Tom', 2: 'Macy', 3: 'Sam'}

# Create dictionary using dict() with keyword arguments as input(when keys are strings).
numbers = dict(one=1, two=2, three=3)
print(numbers)  # Output: {'one': 1, 'two': 2, 'three': 3}

# Dictionary keys have to be unique.
employees = {1: 'Tom', 1: 'Macy'}
# Note that the second assignment overwrites the value associated with the key.
print(employees)  # Output: {1: 'Macy'}

# Dictionary keys and values can be of different data types.
employees = {1: 'Tom', 'Macy': 2, (1, '1'): 'Sam'}
print(employees)  # Output: {1: 'Tom', 'Macy': 2, (1, '1'): 'Sam'}

# Length of a dictionary, or number of key-value pairs in a dictionary.
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(len(employees))  # Output: 3

# Check if a key is present in a dictionary using 'in'.
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(1 in employees)  # Output: True
print(5 in employees)  # Output: False
print(1 not in employees)  # Output: False
print(5 not in employees)  # Output: True

# Access elements in a dictionary by indexing using the key.
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(employees[1])  # Output: 'Tom'
# If key is not present, it raises a KeyError.
print(employees[5])  # Error -> KeyError: 5

# Get elements in a dictionary using get().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(employees.get(1))  # Output: 'Tom'
# If key is not present, the retrn value defaults to None, so it doesn't raise a KeyError.
print(employees.get(5))  # Output: None
# We can specify default values if key is not present.
print(employees.get(5, 'Unknown'))  # Output: 'Unknown'

# Add items(key-value pairs) to a dictionary by indexing using key.
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
employees[4] = 'Lucy'
print(employees)  # Output: {1: 'Tom', 2: 'Macy', 3: 'Sam', 4: 'Lucy'}

# Remove items from a dictionary.
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
del employees[3]
print(employees)  # Output: {1: 'Tom', 2: 'Macy'}

# Remove items and get its value using pop().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(employees.pop(3))  # Output: 'Sam'
print(employees)  # Output: {1: 'Tom', 2: 'Macy'}

# Remove items and get the item using popitem().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(employees.popitem())  # Output: (3, 'Sam') <- Note: popitem() follows LIFO
print(employees)  # Output: {1: 'Tom', 2: 'Macy'}

# Update items in a dictionary.
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
employees[1] = 'Max'
print(employees)  # Output: {1: 'Max', 2: 'Macy', 3: 'Sam'}

# Get or add an item using setdefault().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
# If key is present, its value is returned without modifying.
print(employees.setdefault(1, 'Max'))  # Output: 'Tom'
print(employees)  # Output: {1: 'Tom', 2: 'Macy', 3: 'Sam'}
# If key is not present, it is added to the dictionary.
print(employees.setdefault(4, 'Lucy'))  # Output: 'Lucy'
print(employees)  # Output: {1: 'Max', 2: 'Macy', 3: 'Sam', 4: 'Lucy'}

# Get all items in a dictionary using items().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(employees.items())  # Output: dict_items([(1, 'Tom'), (2, 'Macy'), (3, 'Sam')])

# Get all keys in a dictionary using keys().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(employees.keys())  # Output: dict_keys([1, 2, 3])

# Get all values in a dictionary using values().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
print(employees.values())  # Output: dict_values(['Tom', 'Macy', 'Sam'])

# Get a key iterator for a dictionary using iter().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
employees_key_iterator = iter(employees)
print(employees_key_iterator)  # Output: <dict_keyiterator object at 0x10d8fdea8>
for i in employees_key_iterator:
    print(i)
# Output: 1 2 3

# Remove all items in a dictionary using clear().
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
employees.clear()
print(employees)  # Output: {}

# Delete dictionary using 'del' keyword.
employees = {1: 'Tom', 2: 'Macy', 3: 'Sam'}
del employees

#
# # Here is an example of how to visualize Dictionaries
# print("# Here is an example of how to visualize Dictionaries\n")
#
#
# def print_tree(obj, indent=0):
#     if isinstance(obj, dict):
#         for key, value in obj.items():
#             print(' ' * indent + str(key))
#             print_tree(value, indent + 4)
#     elif isinstance(obj, list):
#         for item in obj:
#             print_tree(item, indent)
#     else:
#         print(' ' * indent + str(obj))
#
#
# company = {
#     "name": "TechCorp",
#     "address": {
#         "street": "123 Main St",
#         "city": "San Francisco",
#         "state": "CA",
#         "zip": "94105"
#     },
#     "employee": [
#         {
#             "id": 1,
#             "name": "Alice",
#             "position": "Software Engineer",
#             "skills": ["Python", "JavaScript", "Git"],
#             "salary": 90000
#         },
#         {
#             "id": 2,
#             "name": "Bob",
#             "position": "Project Manager",
#             "skills": ["Agile", "Scrum", "Kanban"],
#             "salary": 100000
#         },
#         {
#             "id": 3,
#             "name": "Charlie",
#             "position": "Data Scientist",
#             "skills": ["Python", "R", "SQL", "Machine Learning"],
#             "salary": 110000
#         }
#     ],
#     "departments": {
#         "Engineering": {
#             "manager": "Bob",
#             "budget": 1000000
#         },
#         "Data Science": {
#             "manager": "Charlie",
#             "budget": 800000
#         }
#     }
# }
#
# print_tree(company)
