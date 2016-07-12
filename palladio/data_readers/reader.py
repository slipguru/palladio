class Reader(object):
    
    def __init__(self, dataset_files, dataset_options):
        
        self._dataset_files = dataset_files
        self._dataset_options = dataset_options
        
    def get_file(self, file_key):
        
        return self._dataset_files[file_key]
    
    def get_option(self, option_key):
        
        return self._dataset_options[option_key]