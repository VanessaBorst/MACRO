SEED = 123
CUDA_VISIBLE_DEVICES = "MIG-11c29e81-e611-50b5-b5ef-609c0a0fe58b"
TUNE_TEMP_DIR = "/home/vab30xh/"


####### Configuration of Warning Messages #######
import sys
import re
def suppress_warnings():
    """
    Suppress specific warnings
    """
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DeprecationWarning)

        # Add a custom filter to suppress warnings starting with a specific message

        # The following warning is caused since 'use_deterministic_algorithms' conflicts with the entmax
        # activation function and hence, warn_only had to be set to True in 'use_deterministic_algorithms':
        warnings.filterwarnings(action="ignore",
                                category=UserWarning,
                                message=re.escape("cumsum_cuda_kernel does not have a deterministic implementation, but"
                                                  " you set 'torch.use_deterministic_algorithms(True, warn_only=True)'")
                                )

