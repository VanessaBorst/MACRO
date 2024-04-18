SEED = 123
CUDA_VISIBLE_DEVICES = "MIG-11c29e81-e611-50b5-b5ef-609c0a0fe58b"
TUNE_TEMP_DIR = "/home/vab30xh/"


####### Configuration of Warning Messages #######
import sys
def suppress_warnings():
    """
    Suppress specific warnings
    """
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DeprecationWarning)

