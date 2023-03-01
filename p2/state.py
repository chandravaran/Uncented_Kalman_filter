import numpy as np
from quaternion import Quaternion

class FilterState:
    
    def __init__(self, q, w):
        self.quaternion = q
        self.angular_velocity = w

    @classmethod
    def from_vector(cls, x):
        quaternion = Quaternion()
        quaternion.from_axis_angle(np.array(x[:3]))
        angular_velocity = np.array(x[3:6]) 
        return cls(quaternion, angular_velocity)

    def __add__(self, other):

        quaternion = self.quaternion*other.quaternion
        angular_velocity = self.angular_velocity + other.angular_velocity

        return FilterState(quaternion, angular_velocity)
    
    def __sub__(self, other):

        quaternion = self.quaternion*other.quaternion.inv()
        angular_velocity = self.angular_velocity - other.angular_velocity
        return FilterState(quaternion, angular_velocity)