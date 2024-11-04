import random
class Vacuumcleaner:
    def __init__(self, room_size):
        self.room_size=room_size
        self.position=(0,0)
        self.cleaned_rooms=set()
    
    def move(self):
        directions=[(0,1),(0,-1),(1,0),(-1,0)]
        dx, dy=random.choice(directions)
        new_x, new_y=self.position[0] + dx,self.position[1] + dy
        if (0<=new_x < self.room_size) and (0<=new_y < self.room_size):
            self.position=(new_x, new_y)
    
    def clean(self):
        if self.position not in self.cleaned_rooms:
            self.cleaned_rooms.add(self.position)
            print(f"Cleaned room at positions {self.position}")
    
    def run(self):
        while len(self.cleaned_rooms) < self.room_size**2:
            self.move()
            self.clean()
            print(f"Position: {self.position}, Cleaned Rooms: {len(self.cleaned_rooms)}")

agent=Vacuumcleaner(2)
agent.run()
