
import numpy as np
import math 

class ClosetMaterial:

    def __init__(self) -> None:
        self.material_path = '/home/peterjaq/Project/optical_film_toolbox/encode_data.txt'
        self.load_point()

    def closest_mat(self, x, y):

        # distances = {}
        min_distances = 9999
        closest_id = None

        for id, mat in self.points.items():
            distances = math.sqrt((mat[1] - x)**2 + (mat[2] - y) **2)
            if distances < min_distances:
                min_distances = distances
                closest_id = id

        return closest_id

    def load_point(self):

        self.points = {}
        
        with open(file=self.material_path) as mf:
            mfds = mf.readlines()
            for mfd in mfds:
                mfd = mfd.strip().split(' ')
                self.points[mfd[0]] = [mfd[1], float(mfd[2]), float(mfd[3])]


if __name__ == "__main__":
    
    closest_material = ClosetMaterial()

    print(closest_material.closest_mat(0.1, 0.2))

            
            