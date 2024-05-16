from src.pysrc.Base import Transportation

class Airplane(Transportation):
    def __init__(self, distance):
        super().__init__(distance)

    def transport(self):
        print(f"Flying for {self.distance} miles.")
