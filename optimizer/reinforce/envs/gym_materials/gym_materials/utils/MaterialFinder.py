
import numpy as np
import math 

from refractivesqlite import dboperations as DB

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

class MaterialLoader:

    def __init__(self) -> None:

        dbpath = "/home/peterjaq/Project/optical_film_toolbox/db_sqlite/refractive.db"
        self.db = DB.Database(dbpath)

    def load_select_material(self, materials_li, wl_min, wl_max, re=1):
        materials = {}
        for matId in materials_li:
            
            mat = self.db.get_material(matId)

            wl = []
            n = []
            k = []

            for i in range(int(wl_min), int(wl_max), re):
                wl.append(i)
                try:
                    n.append(mat.get_refractiveindex(i))
                except:
                    n.append(0)
                try:
                    k.append(mat.get_extinctioncoefficient(i))
                except:
                    k.append(0)

            matName = mat.get_page_info()['book']
            
            materials[matName] = {'wl':wl, 'n':n, 'k':k}

        return materials

    def load_layers(self, materials_li):
        layers = []
        for matId in materials_li:
            
            mat = self.db.get_material(matId)
            matName = mat.get_page_info()['book']
            layers.append(matName)

        return layers 

if __name__ == "__main__":
    
    closest_material = ClosetMaterial()

    print(closest_material.closest_mat(0.1, 0.2))

            
            