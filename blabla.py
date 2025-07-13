import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class Animal():
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def greet(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old"
    
#Create class Dog and Cat which inherits Animal
class Dog(Animal):
    def greet(self):
        return f"Woof, my name is {self.name} and I am {self.age} years old"
class Cat(Animal):
    def greet(self):
        return f"Meow, my name is {self.name} and I am {self.age} years old"
oreo = Animal("oreo",3)


print(oreo.greet())
