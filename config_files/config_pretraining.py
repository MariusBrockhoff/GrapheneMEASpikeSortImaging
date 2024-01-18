class Config_Pretraining(object):
    """
    Configuration class for the pretraining phase of machine learning models.

    This class initializes and stores configuration parameters necessary for
    the pretraining of machine learning models. It includes settings for data
    preparation, model training, and clustering, tailored to the specific requirements
    of the model and dataset.

    Attributes:
        data_path (str): The file path for the dataset used in pretraining.
        MODEL_TYPE (str): The type of model being pretrained.
        FILE_NAME (str): Extracted file name from `data_path`, used for saving the model.

        # Data Preparation Parameters
        DATA_SAVE_PATH (str): The path where processed data will be saved.
        DATA_PREP_METHOD (str): Method used for data preparation, set to "gradient".
        DATA_NORMALIZATION (str): Type of data normalization to apply, set to "MinMax".
        TRAIN_TEST_SPLIT (float): The ratio of train to test data split.
        BENCHMARK_START_IDX (int): The starting index for benchmarking.
        BENCHMARK_END_IDX (int): The ending index for benchmarking, based on the train-test split ratio.

        # Reconstruction Parameters
        LEARNING_RATE (float): The initial learning rate for model training.
        WITH_WARMUP (bool): Flag to determine if learning rate warmup should be used.
        LR_WARMUP (int): Number of epochs for learning rate warmup.
        LR_FINAL (float): Final learning rate after warmup.
        NUM_EPOCHS (int): Total number of epochs for training.
        BATCH_SIZE (int): Batch size used for training.
        EARLY_STOPPING (bool): Whether to use early stopping.
        PATIENCE (int): Number of epochs to wait for improvement before early stopping.
        MIN_DELTA (float): Minimum change to quantify an improvement.
        BASELINE (int): Baseline value for training metrics.

        # Regularization Parameters
        WITH_WD (bool): Flag to determine if weight decay regularization is used.
        WEIGHT_DECAY (float): Initial value for weight decay.
        WD_FINAL (float): Final value for weight decay after training.

        # Clustering Parameters
        CLUSTERING_METHOD (str): Clustering algorithm to use, set to "Kmeans".
        N_CLUSTERS (int): Number of clusters to form in clustering.
        EPS (float or None): Epsilon parameter for clustering algorithms, if applicable.
        MIN_CLUSTER_SIZE (int): Minimum size for a cluster.
        KNN (int): Number of nearest neighbors to consider in clustering algorithms.

        # Model Saving Parameters
        SAVE_WEIGHTS (bool): Whether to save the model weights after training.
        SAVE_DIR (str): The directory path to save the pretrained model.
    """

    def __init__(self, data_path, model_type):
        """
        The constructor for Config_Pretraining class.

        Initializes the configuration parameters for the pretraining process of
        machine learning models. It sets the paths for data, and defines settings
        for model training, clustering, and saving.

        Parameters:
            data_path (str): The file path for the dataset used in pretraining.
            model_type (str): The type of model being pretrained.
        """

        # Initialize the parent class (object, in this case)
        super(Config_Pretraining, self).__init__()

        # Data and Model Configuration
        self.data_path = data_path  # Path to the dataset
        self.MODEL_TYPE = model_type  # Type of the model
        self.FILE_NAME = self.data_path.rpartition('/')[-1][:-4]  # Extracting file name from the data path

        # Data Preparation Configuration
        self.DATA_SAVE_PATH = self.data_path  # Path for saving processed data
        self.DATA_PREP_METHOD = "gradient"  # Method for data preparation
        self.DATA_NORMALIZATION = "MinMax"  # Normalization technique for data
        self.TRAIN_TEST_SPLIT = 0.75  # Ratio for splitting data into training and testing sets
        self.BENCHMARK_START_IDX = 0  # Starting index for benchmarking
        self.BENCHMARK_END_IDX = 5  # Ending index for benchmarking

        # Reconstruction (Training) Configuration
        self.LEARNING_RATE = 1e-3  # Initial learning rate for training
        self.WITH_WARMUP = False  # Flag to use learning rate warmup
        self.LR_WARMUP = 10  # Number of epochs for learning rate warmup
        self.LR_FINAL = 1e-4  # Final learning rate after warmup
        self.NUM_EPOCHS = 100  # Total number of epochs for training
        self.BATCH_SIZE = 4096  # Batch size for training
        self.EARLY_STOPPING = True  # Use early stopping in training
        self.PATIENCE = 10  # Patience for early stopping
        self.MIN_DELTA = 0.0001  # Minimum delta for improvement in early stopping
        self.BASELINE = 0  # Baseline metric for training

        # Weight Decay (Regularization) Configuration
        self.WITH_WD = False  # Flag to use weight decay
        self.WEIGHT_DECAY = 1e-2  # Initial weight decay value
        self.WD_FINAL = 1e-4  # Final weight decay value after training

        # Clustering Configuration
        self.CLUSTERING_METHOD = "Kmeans"  # Clustering method to be used
        self.N_CLUSTERS = 5  # Number of clusters for Kmeans
        self.EPS = None  # Epsilon parameter for clustering (if applicable)
        self.MIN_CLUSTER_SIZE = 1000  # Minimum size of clusters
        self.KNN = 1000  # Number of nearest neighbors for KNN-based methods

        # Model Saving Configuration
        self.SAVE_WEIGHTS = True  # Flag to save model weights
        self.SAVE_DIR = ("/rds/user/mb2315/hpc-work/Data/Saved_Models/" +
                         "Pretrain_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5")  # Directory for saving the model

