import os
import sys

import re

import yaml 
import numpy as np
import pandas as pd
import math

from tqdm import tqdm 

from scipy import interpolate

sys.path.append("..")

class MaterialLoader:

    def __init__(self, 
                wl_s=0.2, 
                wl_e=2.0):    
        super().__init__()

        self.db_path = './db'
        self.db_info = self.load_db_info()

        self.default_wavelength_range = (wl_s, wl_e)

        self.db_shelf = dict()
        
        self.failed_material = []
        self.success_material = []

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

    def load_total_material(self):
        # print(len(self.material_list))
        # print(type(self.material_list))

        total_material = {}
        for material_name, material_info in self.material_list[1].items():

            try:
                material_path = material_info['path']
                # print(material_path)
                wl, n, k= self.load_material_parameter(material_path)
                # print(material)
                # material_info = self.material 
                total_material[material_name] = {
                    'wl': wl,
                    'n': n,
                    'k': k
                }
                self.success_material.append(material_name)

            except ValueError as ve:
                self.failed_material.append(material_name)
                print('Load %s filled' % material_name)
                print('Material wavelength bound is out of range!')

            except MemoryError as Me:
                self.failed_material.append(material_name)

                print('Load %s filled!' % material_name)
                print('Material wavelength outof memory!')

        # print(total_material, len(total_material))
        return total_material

    def load_total_material_generator(self):


        for material_name, material_info in tqdm(self.material_list[1].items()):

            try:
                # print(material_name)
                material_path = material_info['path']
                
                wl, n, k= self.load_material_parameter(material_path)
            
                self.success_material.append(material_name)
                yield material_name, [wl, n, k]

            except ValueError as ve:

                self.failed_material.append(material_name)
                print('Load %s filled' % material_name)
                print('Material wavelength bound is out of range!')

            except MemoryError as Me:
                self.failed_material.append(material_name)

                print('Load %s filled!' % material_name)
                print('Material wavelength outof memory!')

    def load_select_material(self, select_material):
        selected_material = {}

        for material in select_material:
            
            material_info = self.material_list[1][material]
            wl, n, k= self.load_material_parameter(material_info['path'])

            selected_material[material] = {
                'wl': wl,
                'n': n,
                'k': k
            }

        return selected_material

    def extract_data_nk(self, datas):

        datas_type = datas['DATA'][0]['type']
        wl = []
        n = []
        k = []
        if datas_type == 'tabulated nk':
            datas = datas['DATA'][0]['data'].split('\n')

            for data in datas:
                data = data.strip().split(' ')
                if len(data) == 3:
                        wl.append(float(data[0]) * 1000)
                        n.append(float(data[1]))
                        k.append(float(data[2]))

        elif datas_type == 'tabulated n':
            datas = datas['DATA'][0]['data'].split('\n')

            for data in datas:
                data = data.split(' ')
                wl.append(float(data[0]) * 1000)
                n.append(float(data[1]))
                k.append(0)
        
        elif datas_type == 'tabulated k':
            datas = datas['DATA'][0]['data'].split('\n')

            for data in datas:
                data = data.split(' ')
                wl.append(float(data[0]) * 1000)
                n.append(0)
                k.append(float(data[1]))

        elif datas_type.split(' ')[0] == 'formula':
            coefficients = list(map(float, datas['DATA'][0]['coefficients'].split(' ')))
            wavelength_range = list(map(float, datas['DATA'][0]['wavelength_range'].split(' ')))
            print(wavelength_range)
            wl_tmp = list(np.arange(wavelength_range), 0.001)
            wl = [1000*w for w in wl_tmp]

            if datas_type == 'formula 1':
                n = [self.formula_1(w, coefficients) for w in wl_tmp]
            elif datas_type == 'formula 2':
                n = [self.cauchy_model(w, coefficients) for w in wl_tmp]
            elif datas_type == 'formula 4':
                n = [self.formula_4(w, coefficients) for w in wl_tmp]
            elif datas_type == 'formula 5':
                n = [self.formula_5(w, coefficients) for w in wl_tmp]
            elif datas_type == 'formula 6':
                n = [self.formula_6(w, coefficients) for w in wl_tmp]
            elif datas_type == 'formula 8':
                n = [self.formula_8(w, coefficients) for w in wl_tmp]
            k = [0 for x in range(len(wl))]
            coefficients = list(map(float, datas['DATA'][0]['coefficients'].split(' ')))
            wavelength_range = list(map(float, datas['DATA'][0]['wavelength_range'].split(' ')))

        fwl = np.arange(math.ceil(min(wl)), int(max(wl)), 1)

        fn = interpolate.interp1d(np.array(wl), np.array(n), kind='quadratic')
        fk = interpolate.interp1d(np.array(wl), np.array(k), kind='quadratic')

        return fwl, fn(fwl), fk(fwl)


    def load_material_parameter(self, path):
        
        fileYml = open(os.path.join(self.db_path, 'data', path), encoding='UTF-8')
        datas = yaml.load(fileYml)
        if len(datas['DATA']) == 1:
            wl, n, k = self.extract_data_nk(datas)
        
        elif len(datas['DATA']) == 2:
            wl, n, k = self.extract_data_nk(datas)

        return wl, n, k 

    def formula_1(self, wavelength, coefficients):
        """[summary]
        
        Arguments:
            wavelength {[type]} -- [description]
            coefficients {[type]} -- [description]
        """
        wavelength_square = pow(wavelength, 2)

        if len(coefficients) == 3:
            n_square = 1 + coefficients[0] \
                    + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2)))

        elif len(coefficients) == 5:
            n_square = 1 + coefficients[0] \
                    + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2)))\
                    + ((coefficients[3] * wavelength_square)/(wavelength_square - pow(coefficients[4], 2)))

        elif len(coefficients) == 7:
            n_square = 1 + coefficients[0] \
                    + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2)))\
                    + ((coefficients[3] * wavelength_square)/(wavelength_square - pow(coefficients[4], 2)))\
                    + ((coefficients[5] * wavelength_square)/(wavelength_square - pow(coefficients[6], 2))) 
        
        elif len(coefficients) == 9:
            n_square = 1 + coefficients[0] \
                    + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2)))\
                    + ((coefficients[3] * wavelength_square)/(wavelength_square - pow(coefficients[4], 2)))\
                    + ((coefficients[5] * wavelength_square)/(wavelength_square - pow(coefficients[6], 2)))\
                    + ((coefficients[7] * wavelength_square)/(wavelength_square - pow(coefficients[8], 2)))

        elif len(coefficients) == 11:
            n_square = 1 + coefficients[0] \
                    + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2))) \
                    + ((coefficients[3] * wavelength_square)/(wavelength_square - pow(coefficients[4], 2))) \
                    + ((coefficients[5] * wavelength_square)/(wavelength_square - pow(coefficients[6], 2))) \
                    + ((coefficients[7] * wavelength_square)/(wavelength_square - pow(coefficients[8], 2))) \
                    + ((coefficients[9] * wavelength_square)/(wavelength_square - pow(coefficients[10], 2)))

        elif len(coefficients) == 13:
            n_square = 1 + coefficients[0] \
                    + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2))) \
                    + ((coefficients[3] * wavelength_square)/(wavelength_square - pow(coefficients[4], 2))) \
                    + ((coefficients[5] * wavelength_square)/(wavelength_square - pow(coefficients[6], 2))) \
                    + ((coefficients[7] * wavelength_square)/(wavelength_square - pow(coefficients[8], 2))) \
                    + ((coefficients[9] * wavelength_square)/(wavelength_square - pow(coefficients[10], 2))) \
                    + ((coefficients[11] * wavelength_square)/(wavelength_square - pow(coefficients[12], 2))) 

        elif len(coefficients) == 15:
            n_square = 1 + coefficients[0] \
                    + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2))) \
                    + ((coefficients[3] * wavelength_square)/(wavelength_square - pow(coefficients[4], 2))) \
                    + ((coefficients[5] * wavelength_square)/(wavelength_square - pow(coefficients[6], 2))) \
                    + ((coefficients[7] * wavelength_square)/(wavelength_square - pow(coefficients[8], 2))) \
                    + ((coefficients[9] * wavelength_square)/(wavelength_square - pow(coefficients[10], 2))) \
                    + ((coefficients[11] * wavelength_square)/(wavelength_square - pow(coefficients[12], 2))) \
                    + ((coefficients[13] * wavelength_square)/(wavelength_square - pow(coefficients[14], 2))) 

        elif len(coefficients) == 17:
            n_square = 1 + coefficients[0] \
                    + ((coefficients[1] * wavelength_square)/(wavelength_square - pow(coefficients[2], 2))) \
                    + ((coefficients[3] * wavelength_square)/(wavelength_square - pow(coefficients[4], 2))) \
                    + ((coefficients[5] * wavelength_square)/(wavelength_square - pow(coefficients[6], 2))) \
                    + ((coefficients[7] * wavelength_square)/(wavelength_square - pow(coefficients[8], 2))) \
                    + ((coefficients[9] * wavelength_square)/(wavelength_square - pow(coefficients[10], 2))) \
                    + ((coefficients[11] * wavelength_square)/(wavelength_square - pow(coefficients[12], 2))) \
                    + ((coefficients[13] * wavelength_square)/(wavelength_square - pow(coefficients[14], 2))) \
                    + ((coefficients[15] * wavelength_square)/(wavelength_square - pow(coefficients[16], 2))) \

        return math.sqrt(n_square)

    def formula_4(self, wavelength, coefficients):
        wavelength_square = pow(wavelength, 2)
        if len(coefficients) == 9:
            n_square = coefficients[0]
            n_square += (coefficients[1] * pow(wavelength, coefficients[2])) / (wavelength_square - pow(coefficients[3], coefficients[4]))
            n_square += (coefficients[5] * pow(wavelength, coefficients[6])) / (wavelength_square -  pow(coefficients[7], coefficients[8]))

        elif len(coefficients) == 11:
            n_square = coefficients[0]
            n_square += (coefficients[1] * pow(wavelength, coefficients[2])) / (wavelength_square - pow(coefficients[3], coefficients[4]))
            n_square += (coefficients[5] * pow(wavelength, coefficients[6])) / (wavelength_square -  pow(coefficients[7], coefficients[8]))
            n_square += coefficients[9] * pow(wavelength, coefficients[10])

        elif len(coefficients) == 13:
            n_square = coefficients[0]
            n_square += (coefficients[1] * pow(wavelength, coefficients[2])) / (wavelength_square - pow(coefficients[3], coefficients[4]))
            n_square += (coefficients[5] * pow(wavelength, coefficients[6])) / (wavelength_square -  pow(coefficients[7], coefficients[8]))
            n_square += coefficients[9] * pow(wavelength, coefficients[10])
            n_square += coefficients[11] * pow(wavelength, coefficients[12])
                
        elif len(coefficients) == 15:
            n_square = coefficients[0]
            n_square += (coefficients[1] * pow(wavelength, coefficients[2])) / (wavelength_square - pow(coefficients[3], coefficients[4]))
            n_square += (coefficients[5] * pow(wavelength, coefficients[6])) / (wavelength_square -  pow(coefficients[7], coefficients[8]))
            n_square += coefficients[9] * pow(wavelength, coefficients[10])
            n_square += coefficients[11] * pow(wavelength, coefficients[12])
            n_square += coefficients[13] * pow(wavelength, coefficients[14])

        return math.sqrt(n_square)

    def formula_5(self, wavelength, coefficients):
        n_square = 1 
        if len(coefficients) == 3:
            n_square += coefficients[0] + coefficients[1]*pow(wavelength, coefficients[2])

        return math.sqrt(n_square)

    def formula_6(self, wavelength, coefficients):
        wavelength_sqrt = pow(wavelength, -2)

        if len(coefficients) == 11:
            n = 1 + coefficients[0] \
                + (coefficients[1]/(coefficients[2] - wavelength_sqrt)) \
                + (coefficients[3]/(coefficients[4] - wavelength_sqrt)) \
                + (coefficients[5]/(coefficients[6] - wavelength_sqrt)) \
                + (coefficients[7]/(coefficients[8] - wavelength_sqrt)) \
                + (coefficients[9]/(coefficients[10] - wavelength_sqrt))


        elif len(coefficients) == 9:
            n = 1 + coefficients[0] \
                + (coefficients[1]/(coefficients[2] - wavelength_sqrt)) \
                + (coefficients[3]/(coefficients[4] - wavelength_sqrt)) \
                + (coefficients[5]/(coefficients[6] - wavelength_sqrt)) \
                + (coefficients[7]/(coefficients[8] - wavelength_sqrt))


        elif len(coefficients) == 7:
            n = 1 + coefficients[0] \
                + (coefficients[1]/(coefficients[2] - wavelength_sqrt)) \
                + (coefficients[3]/(coefficients[4] - wavelength_sqrt)) \
                + (coefficients[5]/(coefficients[6] - wavelength_sqrt))

        elif len(coefficients) == 5:
            n = 1 + coefficients[0] \
                + (coefficients[1]/(coefficients[2] - wavelength_sqrt)) \
                + (coefficients[3]/(coefficients[4] - wavelength_sqrt))

        elif len(coefficients) == 3:
            n = 1 + coefficients[0] \
                + (coefficients[1]/(coefficients[2] - wavelength_sqrt))

        return n

    def formula_8(self, wavelength, coefficients):
        pass 

    def cauchy_model(self, wavelength, coefficients):
        """[cauchy model]
        
        Arguments:
            wavelength {[type]} -- [description]
            coefficients {[type]} -- [description]
        """

        if len(coefficients)  == 3:
            n = coefficients[0] \
                + (coefficients[1] * pow(wavelength, coefficients[2]))    

        elif len(coefficients)  == 5:
            n = coefficients[0] \
                + (coefficients[1] * pow(wavelength, coefficients[2])) \
                + (coefficients[3] * pow(wavelength, coefficients[4]))

        elif len(coefficients) == 7:

            n = coefficients[0] \
                + (coefficients[1] * pow(wavelength, coefficients[2])) \
                + (coefficients[3] * pow(wavelength, coefficients[4])) \
                + (coefficients[5] * pow(wavelength, coefficients[6]))

        elif len(coefficients) == 9:

            n = coefficients[0] \
                + (coefficients[1] * pow(wavelength, coefficients[2])) \
                + (coefficients[3] * pow(wavelength, coefficients[4])) \
                + (coefficients[5] * pow(wavelength, coefficients[6])) \
                + (coefficients[7] * pow(wavelength, coefficients[8]))

        elif len(coefficients) == 11:

            n = coefficients[0] \
                + (coefficients[1] * pow(wavelength, coefficients[2])) \
                + (coefficients[3] * pow(wavelength, coefficients[4])) \
                + (coefficients[5] * pow(wavelength, coefficients[6])) \
                + (coefficients[7] * pow(wavelength, coefficients[8])) \
                + (coefficients[9] * pow(wavelength, coefficients[10]))    

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
    # ml.load_total_material()


    for sample in ml.load_total_material_generator():
        print(sample.keys())

    print("Success Load %s materials." % len(ml.success_material))
    print("Failed Load %s materials." % len(ml.failed_material))
    for failed in ml.failed_material:
        print("Failed Load %s " % failed)