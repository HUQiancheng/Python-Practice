# Modifying
auto = ['door', 'wheel', 'steer', 'light', 'engine', 'seat']
print(auto)
auto[-2] = 'engineer'
print(auto)

# Adding Elements to a list
auto.append('engine')
print(auto)
auto = []
print(auto)
auto.append('door')
auto.append('wheel')
auto.append('steer')
auto.append('light')
auto.append('seat')
print(auto)

# Inserting elements into a list..........Page 45
auto.insert(4, 'engine')
print(auto)
auto.insert(5, 'baga!')
print(auto)

# Removing Elements using del, .pop() and remove()

# del
del auto[5]
print(auto)
# pop
popped_auto_end = auto.pop()
popped_auto_2 = auto.pop(2)
print(auto)
print(popped_auto_end)
print(popped_auto_2)
# remove
auto.insert(-1, "seat")
auto.insert(2, "steer")
print(auto)
auto.remove("light")
