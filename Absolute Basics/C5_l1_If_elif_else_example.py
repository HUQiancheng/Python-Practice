# for,%,if
autos = ["audi", "bmw", "subaru", "toyota", "benz", "hammer", "jeep", "kia"]
for i in range(0, 8):
    if (i % 3) == 1:
        print(autos[i].upper())
    elif (i % 2) == 0:
        print(autos[i].lower())
    else:
        print("Hello world!")

# Check if value is in a list
if (("audi" in autos) or ("toyota" not in autos)) and "bmw" in autos:
    print("You got me!")
