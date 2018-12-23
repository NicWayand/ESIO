import os
import pickle


class EsioData(object):
    """ ESIO storage of data path and info. """

    def __init__(self, data_dir=None):
        """"""
        self.data_dir = data_dir
        self.obs_dir = os.path.join(self.data_dir, 'obs')
        self.model_dir = os.path.join(self.data_dir, 'model')
        self.grid_dir = os.path.join(self.data_dir, 'grids')
        self.make_dir(os.path.join(data_dir, 'grids'))

        self.obs = {} #os.path.join(data_dir, 'obs')
        self.model = {} #os.path.join(data_dir, 'model')
        
    def make_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def add_obs(self, obs_name, grid_file=None):
        """Add an obs"""
        self.obs[obs_name] =  {'native':os.path.join(self.obs_dir, obs_name, 'native'),
                               'sipn_nc':os.path.join(self.obs_dir, obs_name, 'sipn_nc')}
        
        # Add grid file
        self.obs[obs_name]['grid'] = os.path.join(self.grid_dir, grid_file)
        
    def add_model(self, mod_name=None, model_label=None, grid_file=None):
        """Add a model"""
        mod_dir = os.path.join(self.data_dir, 'model', mod_name)
        self.make_dir(mod_dir)
        
        # Loop through new dirs
        new_dirs = ['forecast','reanalysis','reforecast']
        new_s_dirs = ['native','sipn_nc','sipn_nc_agg']
        nd_dict = {}
        for nd in new_dirs:
            # Create it
            self.make_dir(os.path.join(mod_dir, nd))
            nsd_dict = {}       
            for nsd in new_s_dirs:
                # Make new dir string
                cdir = os.path.join(mod_dir, nd, nsd)
                # Create it
                self.make_dir(cdir)
                # Add it
                nsd_dict[nsd] = cdir
            nd_dict[nd] = nsd_dict
        
        # Add built up dict
        self.model[mod_name] =  nd_dict   
        
        # Add grid file
        self.model[mod_name]['grid'] = os.path.join(self.grid_dir, grid_file)
        
        # Add model label name
        self.model[mod_name]['model_label'] = model_label
        

        
    def save(self, filename='ESIO_DATA.pkl'):
        # Get dir from env variabel DATA_DIR
        DATA_DIR = os.getenv('DATA_DIR')
        if not DATA_DIR:
            raise ValueError("Env variable DATA_DIR not set")
        with open(os.path.join(DATA_DIR, filename), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)    
            
    def load(filename='ESIO_DATA.pkl'):
        # Get dir from env variabel DATA_DIR
        DATA_DIR = os.getenv('DATA_DIR')
        if not DATA_DIR:
            raise ValueError("Env variable DATA_DIR not set")
        return pickle.load( open( os.path.join(DATA_DIR, filename), 'rb' ) )

