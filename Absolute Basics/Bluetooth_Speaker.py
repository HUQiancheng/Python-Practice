class EssentialSpeaker:
    def __init__(self, slider, button, plastic, metal, connections, chip, factory, sd_card_memory):
        self.slider = slider
        self.button = button
        self.plastic = plastic
        self.metal = metal
        self.connections = connections
        self.chip = chip
        self.factory = factory
        self.sd_card_memory = sd_card_memory

    def play_music(self):
        return f"Playing music on an Essential Speaker made in factory {self.factory}."

    def volume_adjustment(self):
        return f"Volume adjustment using slider {self.slider} and button {self.button}."

    def sd_card_usage(self):
        return f"SD card memory: {self.sd_card_memory} GB."


class ProfessionalSpeaker(EssentialSpeaker):
    def __init__(self, slider, button, plastic, metal, connections, chip, factory, sd_card_memory, song_switch, bluetooth, lightweight_material):
        super().__init__(slider, button, plastic, metal, connections, chip, factory, sd_card_memory)
        self.song_switch = song_switch
        self.bluetooth = bluetooth
        self.lightweight_material = lightweight_material

    def song_switching(self):
        return f"Song switching using {self.song_switch}."

    def bluetooth_connection(self):
        return f"Bluetooth connection supported: {self.bluetooth}."

    def lightweight_material_usage(self):
        return f"Lightweight material used: {self.lightweight_material}."


# Instantiate objects
essential_speaker = EssentialSpeaker('Sony', 'Samsung', 'PVC', 'Aluminum', 'Gold-plated', 'Qualcomm', 1, 32)
professional_speaker = ProfessionalSpeaker('Sony', 'Samsung', 'PVC', 'Aluminum', 'Gold-plated', 'Qualcomm', 1, 64, 'touchpad', True, 'Carbon fiber')

# Test methods for the Essential Speaker
print(essential_speaker.play_music())
print(essential_speaker.volume_adjustment())
print(essential_speaker.sd_card_usage())

# Test methods for the Professional Speaker
print(professional_speaker.play_music())
print(professional_speaker.volume_adjustment())
print(professional_speaker.sd_card_usage())
print(professional_speaker.song_switching())
print(professional_speaker.bluetooth_connection())
print(professional_speaker.lightweight_material_usage())

# Explanation:
#
# EssentialSpeaker class represents the blueprint for the essential speaker with attributes such as slider, button,
# plastic, metal, connections, chip, factory, and sd_card_memory. It has methods for playing music, volume adjustment,
# and SD card memory usage.
# ProfessionalSpeaker class inherits from the EssentialSpeaker class and adds additional attributes like song_switch,
# bluetooth, and lightweight_material. It also has methods for song switching, Bluetooth connection, and lightweight
# material usage.
# The super() function is used to call the parent class's __init__ method, ensuring that the attributes of the parent
# class are properly initialized.
# Instances of both the EssentialSpeaker and ProfessionalSpeaker classes are created and tested to demonstrate the
# functionality of the code.
