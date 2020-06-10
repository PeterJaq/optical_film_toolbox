import configparser
import os 
import sys 

import ast 


class FilmConfig:

    def __init__(self, config_root_path='./config', config_path='Zn.ini'):
        super().__init__()
        self.config = configparser.ConfigParser()

        self.material_config, self.parameter_config, self.target_weight, self.agent = self.load_config(config_root_path,
                                                                                           config_path)

    def load_config(self, config_root_path, config_path):
        self.config.read(os.path.join(config_root_path, config_path))

        return self.config['Material'], self.config['Parameter'], self.config['Target and Weight'], self.config['Agent']

    @property
    def materials(self):
        return ast.literal_eval(self.material_config['materials'])

    @property
    def WLrange(self):

        return ast.literal_eval(self.parameter_config['WLrange'])

    @property
    def round_threshold(self):

        return float(self.agent['round_threshold'])

    @property
    def end_threshold(self):

        return float(self.agent['end_threshold'])

    @property
    def init_state(self):

        return ast.literal_eval(self.agent['init_state']) 

    @property
    def WLstep(self):

        return float(self.parameter_config['WLstep'])

    @property
    def targets(self):

        return ast.literal_eval(self.target_weight['targets'])

    @property
    def weights(self):

        return ast.literal_eval(self.target_weight['weights'])



if __name__ == "__main__":
    
    config_path = 'Zn.ini'
    fmconf = FilmConfig()
    #material, parameters = fmconf.conf.load_config(config_path)

    """
    Test Config
    """

    print(fmconf.materials)
    print(fmconf.WLstep)
    print(fmconf.WLrange)
    print(fmconf.target)
    print(fmconf.weight)