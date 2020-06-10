import os
import sys

import re

import yaml 
import numpy as np
import pandas as pd
import math

from scipy import interpolate

sys.path.append("..")

class MaterialLoader:

    def __init__(self):
        super().__init__()

        self.db_path = './db'
        self.db_info = self.load_db_info()

        self.db_shelf = dict()

    def load_db_info(self):
        """[summery]
        load data base info from database/library.yml

        Returns:
            [db info] -- [total information from load from library.yml]
        """

        info_path = 'library.yml'
        
        fileYml = open(os.path.join(self.db_path,info_path), encoding='UTF-8')
        db_info = yaml.load(fileYml)
    
        return db_info

    def load_material(self, shelfs):
        """[summary]
        using in self.material_list()

        load material data path from db_info
        Arguments:
            shelfs {[str]} -- [shelfs name list]
        
        Returns:
            [material_names, material_data] -- [material_data]
        """
        material_names = []
        material_data = {}

        for shelf in shelfs:
            for material in self.db_shelf[shelf]:
                if 'BOOK' in material.keys():
                    material_names.append(material['BOOK'])
                    for data in material['content']:
                        
                        if 'data' in data.keys():
                            material_data['%s_%s'%(material['BOOK'], data['PAGE'])] = self.material_info_split(divider, data['name'])
                            material_data['%s_%s'%(material['BOOK'], data['PAGE'])]['path'] = data['data']
                            material_data['%s_%s'%(material['BOOK'], data['PAGE'])]['divider'] = divider

                        elif 'DIVIDER' in data.keys():
                            divider = data['DIVIDER']
        
        return material_names, material_data

    def material_info_split(self, divider, info):
        material_info = {}

        info_split = info.split(':')
        if len(info_split) > 1:
            material_info['year'], material_info['author'] = self.rex_author_info(info_split[0])
            material_info['n'], material_info['k'] = self.rex_nk_info(info_split[1])
            material_info['wavelength_start'], material_info['wavelength_end'] = self.rex_wavelength_info(info_split[1])
            material_info['degree'] = self.rex_degree_info(info_split[1])
            material_info['model'] = self.rex_model_info(info_split[1])
        else:
            material_info['year'], material_info['author'] = self.rex_author_info(info_split[0])
            material_info['n'], material_info['k'] = True, False
            material_info['wavelength_start'], material_info['wavelength_end'] = None, None
            material_info['degree'] = None
            material_info['model'] = None
        return material_info

    def rex_author_info(self, info):

        try:
            year = re.findall('[0-9]{4}', info)[0]
            author = info.split(year)[0]
        except:
            year = None
            author = info

        return year, author

    def rex_nk_info(self, info):
        try:
            nk = re.findall('n,k', info)
            if nk is not None:
                n = True 
                k = True
            else:
                n = True
                k = False
        except:
            n = False
            k = False
            return n, k

        return n, k

    def rex_wavelength_info(self, info):
        try:
            wavelength_range = re.findall('-?\d+\.\d*\d*?',info)
            if len(wavelength_range) is 2:
                wavelength_start, wavelength_end = wavelength_range[0], wavelength_range[1]
            else:
                wavelength_start = wavelength_range[0]
                wavelength_end = wavelength_range[0]
        except:
            wavelength_start = None
            wavelength_end = None

        return wavelength_start, wavelength_end

    def rex_degree_info(self, info):
        degree = re.findall('\-?\d+?°C', info)
        if len(degree) == 0:
            return None
        
        return degree[0]

    def rex_model_info(self, info):
        model = re.findall('Brendel-Bormann model', info)
        if len(model) != 0:
            return 'Brendel-Bormann model'
        
        model = re.findall('Lorentz-Drude model', info)
        if len(model) != 0:
            return 'Lorentz-Drude model'

        model = re.findall('DFT calculations', info)
        if len(model) != 0:
            return 'DFT calculations'
        
        return None

    def load_select_material(self, select_material):
        selected_material = {}

        for material in select_material:

            material_info = self.material_list[1][material]
            material_path = material_info['path']
            wl, n, k= self.load_material_parameter(material_path)

            selected_material[material] = {
                'wl': wl,
                'n': n,
                'k': k
            }

        return selected_material

    def load_material_parameter(self, path):
        
        fileYml = open(os.path.join(self.db_path, 'data', path), encoding='UTF-8')
        datas = yaml.load(fileYml)
        if len(datas['DATA']) == 1:
            datas_type = datas['DATA'][0]['type']
            wl = []
            n = []
            k = []
            if datas_type == 'tabulated nk':
                datas = datas['DATA'][0]['data'].split('\n')

                for data in datas:
                    data = data.split(' ')
                    if len(data) == 3:
                        wl.append(float(data[0]) * 1000)
                        n.append(float(data[1]))
                        k.append(float(data[2]))

            elif datas_type == 'tabulated n':
                datas = datas['DATA'][0]['data'].split('\n')

                for data in datas:
                    data = data.split(' ')
                    if len(data) == 2:
                        wl.append(float(data[0]) * 1000)
                        n.append(float(data[1]))
                        k.append(0)

            elif datas_type == 'formula 1':
                coefficients = list(map(float, datas['DATA'][0]['coefficients'].split(' ')))
                wavelength_range = list(map(float, datas['DATA'][0]['wavelength_range'].split(' ')))

                wl = list(np.arange(wavelength_range[0] * 1000, wavelength_range[1] * 1000, 1))

                """Cauchy model"""
                #wl = list(np.arange(min(wavelength_range[0]), max(wavelength_range[1]), 0.001))

                n = [self.formula_1(w, coefficients) for w in wl]
                
                k = [0 for x in range(len(wl))]

            fwl = np.arange(min(wl), max(wl), 1)
            fn = interpolate.interp1d(wl, n,  kind='quadratic')
            fk = interpolate.interp1d(wl, k, kind='quadratic')

            return fwl, fn(fwl), fk(fwl)

        elif len(datas['DATA'] == 2):
            wl = []
            n = []
            k = []

            max_wl = 0
            min_wl = 99999

            datas = datas['DATA']
            for data in datas:
                if datas['type'] == 'tabulated k':
                    wl_k = datas['data'].split('\n')
                    if len(data) == 2:
                        wl.append(float(wl_k[0]) * 1000)
                        k.append(float(wl_k[1]))

                    max_wl = max(max_wl, max(wl))
                    min_wl = min(min_wl, min(wl))

                elif datas['type'] == 'tabulated n':
                    wl_n = datas['data'].split('\n')
                    if len(data) == 2:
                        wl.append(float(wl_n[0]) * 1000)
                        n.append(float(wl_n[1]))

                    max_wl = max(max_wl, max(wl))
                    min_wl = min(min_wl, min(wl))

                elif datas['type'] == 'formula 2':
                    coefficients = list(map(float, datas['DATA'][0]['coefficients'].split(' ')))
                    wavelength_range = list(map(float, datas['DATA'][0]['wavelength_range'].split(' ')))

                    max_wl = max(max_wl, wavelength_range[1])
                    min_wl = min(min_wl, wavelength_range[0])

                    wl = list(np.arange(wavelength_range[0] * 1000, wavelength_range[1] * 1000), 1)

                    """Cauchy model"""
                    n = [self.cauchy_model(w, coefficients) for w in fwl]

            fwl = np.arange(min_wl, max_wl, 1)
            fn = interpolate.interp1d(wl, n)
            fk = interpolate.interp1d(wl, k)

            return fwl, fn(fwl), fk(fwl)

    def formula_1(self, wavelength, coefficients):
        """[summary]
        
        Arguments:
            wavelength {[type]} -- [description]
            coefficients {[type]} -- [description]
        """
        wavelength_square = pow(wavelength, 2)
        n_square = 1 + coefficients[0] \
                 + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2)))\
                 + ((coefficients[3] * wavelength_square)/(wavelength_square - pow(coefficients[4], 2)))\
                 + ((coefficients[5] * wavelength_square)/(wavelength_square - pow(coefficients[6], 2)))\

        return math.sqrt(n_square)

    def cauchy_model(self, wavelength, coefficients):
        """[cauchy model]
        
        Arguments:
            wavelength {[type]} -- [description]
            coefficients {[type]} -- [description]
        """
        n = coefficients[0] \
            + (coefficients[1] * pow(wavelength, coefficients[2])) \
            + (coefficients[3] * pow(wavelength, coefficients[4])) \
            + (coefficients[5] * pow(wavelength, coefficients[6]))
            
        return n

    @property
    def shelf(self):
        """[summery]
        load data base shelf from db_info

        Returns:
            [shelf] -- [database shelf]
        """
        shelfs_name = []

        for data in self.db_info:
            self.db_shelf[data['SHELF']] = data['content']
            shelfs_name.append(data['SHELF'])

        return shelfs_name

    @property
    def material_list(self, shelfs='total', shelfs_list=None):
        """[summary]
        
        Keyword Arguments:
            shelfs {str} -- [description] (default: {'total'})
            shelfs_list {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        if shelfs is 'total':
            materials_name, material_data = self.load_material(self.shelf)
        elif shelfs is not 'total':
            materials_name, material_data = self.load_material(shelfs_list)

        return materials_name, material_data


    def load_material_data(self, path):

        fileYml = open(os.path.join(self.db_path, path), encoding='UTF-8')
        db_info = yaml.load(fileYml)


    

        
    
if __name__ == "__main__":

    ml = MaterialLoader()

    print("Load Total %d Material, %d Optical Constants"%(len(ml.material_list[0]), len(ml.material_list[1])))
    print(ml.load_select_material(['Zn_Werner', 'SiO2_Malitson', 'Cu_Johnson']))