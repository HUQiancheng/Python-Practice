# Slicing a List
players = ['charles', 'martina', 'michael', 'florence', 'eli']
print(players[0:3])
print(players[1:4])
print(players[:3])
print(players[2:])
print(players[:])

# Looping Through a Slice
for player in players[2:]:
    print(player.title())

# Copying a List
player01=players[0:1]
player2e=players[2:]
players3=players[:3]

print(player01)
print(player2e)
print(players3)
