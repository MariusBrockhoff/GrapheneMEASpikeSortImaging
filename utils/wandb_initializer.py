import wandb


def wandb_initializer(model_config, pretraining_config, finetune_config, method):
    """
    Initialize a Weights & Biases (wandb) run for experiment tracking and logging.

    This function sets up a wandb run based on the provided method (reconstruction, DEC, IDEC)
    and configuration parameters. It logs various hyperparameters and metadata for the model training.

    Arguments:
        model_config: Configuration object containing model specifications.
        pretraining_config: Configuration object containing pretraining specifications.
        finetune_config: Configuration object containing finetuning specifications.
        method: String indicating the training method ('reconstruction', 'DEC', or 'IDEC').

    The function initializes a wandb run with a project name and configuration parameters specific to the chosen method.
    The project name and parameters are set based on the model type, method, and configurations provided.

    For 'reconstruction' method:
        - The project is named "Pretraining_{method}_{MODEL_TYPE}".
        - Configurations related to pretraining and model settings are logged.

    For 'DEC' and 'IDEC' methods:
        - The project is named "Finetuning_{method}_{MODEL_TYPE}".
        - Configurations related to finetuning and model settings are logged.

    Returns:
        None. The function initializes a wandb run for logging and tracking purposes.
    """

    if method == "reconstruction":
        if model_config.MODEL_TYPE == "DenseAutoencoder":
            wandb.init(
                        # set the wandb project where this run will be logged
                        project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                        # track hyperparameters and run metadata with wandb.model_config
                        config={"Model": model_config.MODEL_TYPE,
                                "DATA": pretraining_config.FILE_NAME,
                                "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                                "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                                "LEARNING_RATE": pretraining_config.LEARNING_RATE,
                                "WITH_WARMUP": pretraining_config.WITH_WARMUP,
                                "LR_WARMUP": pretraining_config.LR_WARMUP,
                                "LR_FINAL": pretraining_config.LR_FINAL,
                                "NUM_EPOCHS": pretraining_config.NUM_EPOCHS,
                                "BATCH_SIZE": pretraining_config.BATCH_SIZE,
                                "DIMS": model_config.DIMS,
                                "ACT": model_config.ACT,
                                "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                                "N_CLUSTERS": pretraining_config.N_CLUSTERS})

    if method == "DEC":
        wandb.init(
                # set the wandb project where this run will be logged
                project="Finetuning_DEC_" + model_config.MODEL_TYPE,
                # track hyperparameters and run metadata with wandb.model_config
                config={"Model": model_config.MODEL_TYPE,
                        "DATA": finetune_config.FILE_NAME,
                        "DEC_N_CLUSTERS": finetune_config.DEC_N_CLUSTERS,
                        "DEC_BATCH_SIZE": finetune_config.DEC_BATCH_SIZE,
                        "DEC_LEARNING_RATE": finetune_config.DEC_LEARNING_RATE,
                        "DEC_MOMENTUM": finetune_config.DEC_MOMENTUM,
                        "DEC_TOL": finetune_config.DEC_TOL,
                        "DEC_MAXITER": finetune_config.DEC_MAXITER,
                        "DEC_UPDATE_INTERVAL": finetune_config.DEC_UPDATE_INTERVAL,
                        "DEC_SAVE_DIR": finetune_config.DEC_SAVE_DIR})

    elif method == "IDEC":
        wandb.init(
                # set the wandb project where this run will be logged
                project="Finetuning_IDEC_" + model_config.MODEL_TYPE,
                # track hyperparameters and run metadata with wandb.model_config
                config={"Model": model_config.MODEL_TYPE,
                        "DATA": finetune_config.FILE_NAME,
                        "IDEC_N_CLUSTERS": finetune_config.IDEC_N_CLUSTERS,
                        "IDEC_BATCH_SIZE": finetune_config.IDEC_BATCH_SIZE,
                        "IDEC_LEARNING_RATE": finetune_config.IDEC_LEARNING_RATE,
                        "IDEC_MOMENTUM": finetune_config.IDEC_MOMENTUM,
                        "IDEC_GAMMA": finetune_config.IDEC_GAMMA,
                        "IDEC_TOL": finetune_config.IDEC_TOL,
                        "IDEC_MAXITER": finetune_config.IDEC_MAXITER,
                        "IDEC_UPDATE_INTERVAL": finetune_config.IDEC_UPDATE_INTERVAL,
                        "IDEC_SAVE_DIR": finetune_config.IDEC_SAVE_DIR})
