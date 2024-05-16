from src.pysrc.Base import Transportation

class Car(Transportation):
    def __init__(self, distance):
        super().__init__(distance) # super() is used to call the parent class's constructor

    def transport(self): # when the name is the same as the parent class's method, it is called method overriding
        print(f"Driving for {self.distance} miles.")

    def play_music(self):
        print("Playing music in the car.")
