class Config_Finetuning(object):
    """
    Configuration class for fine-tuning machine learning models.

    This class is designed to initialize and store configuration parameters
    needed for the fine-tuning of machine learning models. It includes parameters
    specific to the data, pretrained models, and fine-tuning settings for different
    algorithms such as DEC (Deep Embedding Clustering) and IDEC (Improved Deep Embedding Clustering).

    Attributes:
        data_path (str): The file path for the dataset used in fine-tuning.
        MODEL_TYPE (str): The type of model being fine-tuned.
        FILE_NAME (str): Extracted file name from `data_path`, used in naming saved models.
        PRETRAINED_SAVE_DIR (str): The directory path to save or load the pretrained model.
        DEC_N_CLUSTERS (int): The number of clusters for DEC algorithm.
        DEC_BATCH_SIZE (int): Batch size for training in the DEC algorithm.
        DEC_LEARNING_RATE (float): Learning rate for the DEC algorithm.
        DEC_MOMENTUM (float): Momentum for the DEC algorithm.
        DEC_TOL (float): Tolerance for convergence in DEC.
        DEC_MAXITER (int): Maximum number of iterations for DEC.
        DEC_UPDATE_INTERVAL (int): Interval for updating cluster assignments in DEC.
        DEC_SAVE_DIR (str): The directory path to save the DEC model.
        IDEC_N_CLUSTERS (int): The number of clusters for IDEC algorithm.
        IDEC_BATCH_SIZE (int): Batch size for training in the IDEC algorithm.
        IDEC_LEARNING_RATE (float): Learning rate for the IDEC algorithm.
        IDEC_MOMENTUM (float): Momentum for the IDEC algorithm.
        IDEC_GAMMA (float): Coefficient of clustering loss in IDEC.
        IDEC_TOL (float): Tolerance for convergence in IDEC.
        IDEC_MAXITER (int): Maximum number of iterations for IDEC.
        IDEC_UPDATE_INTERVAL (int): Interval for updating cluster assignments in IDEC.
        IDEC_SAVE_DIR (str): The directory path to save the IDEC model.
    """

    def __init__(self, data_path, model_type):
        """
        The constructor for Config_Finetuning class.

        Initializes the configuration parameters for the fine-tuning process of
        machine learning models. It sets the paths for data, pretrained
        models, and the saved models.
        """

        # Initialize the parent class (object, in this case)
        super(Config_Finetuning, self).__init__()

        # Data and Model Configuration
        self.data_path = data_path  # Path to the dataset
        self.MODEL_TYPE = model_type  # Type of the model
        self.FILE_NAME = self.data_path.rpartition('/')[-1][:-4]  # Extracting file name from the data path

        # Pretrained Model Configuration
        self.PRETRAINED_SAVE_DIR = ("/rds/user/mb2315/hpc-work/Data/Saved_Models/" +
                                    "Pretrain_NNCLR_" + self.MODEL_TYPE + "_" +
                                    self.FILE_NAME + ".h5")  # Directory to save/load pretrained model

        # DEC (Deep Embedding Clustering) Configuration
        self.DEC_N_CLUSTERS = 5  # Number of clusters in DEC
        self.DEC_BATCH_SIZE = 256  # Batch size for DEC
        self.DEC_LEARNING_RATE = 0.01  # Learning rate for DEC
        self.DEC_MOMENTUM = 0.9  # Momentum for DEC
        self.DEC_TOL = 0.001  # Tolerance for convergence in DEC
        self.DEC_MAXITER = 8000  # Maximum iterations for DEC
        self.DEC_UPDATE_INTERVAL = 140  # Update interval for DEC
        self.DEC_SAVE_DIR = ("/rds/user/mb2315/hpc-work/Data/Saved_Models/" +
                             "DEC_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5")  # DEC model save directory

        # IDEC (Improved Deep Embedding Clustering) Configuration
        self.IDEC_N_CLUSTERS = 5  # Number of clusters in IDEC
        self.IDEC_BATCH_SIZE = 256  # Batch size for IDEC
        self.IDEC_LEARNING_RATE = 0.1  # Learning rate for IDEC
        self.IDEC_MOMENTUM = 0.99  # Momentum for IDEC
        self.IDEC_GAMMA = 0.1  # Coefficient of clustering loss in IDEC
        self.IDEC_TOL = 0.001  # Tolerance for convergence in IDEC
        self.IDEC_MAXITER = 20000  # Maximum iterations for IDEC
        self.IDEC_UPDATE_INTERVAL = 140  # Update interval for IDEC
        self.IDEC_SAVE_DIR = ("/rds/user/mb2315/hpc-work/Data/Saved_Models/" +
                              "IDEC_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5")  # IDEC model save directory
