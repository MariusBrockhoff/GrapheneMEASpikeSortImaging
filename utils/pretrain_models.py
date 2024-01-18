import tensorflow as tf
import tensorflow_addons as tfa

from os import path
from utils.evaluation import *


def check_filepath_naming(filepath):
    """
    Check and modify the file path to avoid overwriting existing files.

    If the specified file path exists, a number is appended to the file name to
    create a unique file name.

    Arguments:
        filepath: The file path to check.

    Returns:
        A modified file path if the original exists, otherwise the original file path.
    """
    if path.exists(filepath):
        numb = 1
        while True:
            new_path = "{0}_{2}{1}".format(*path.splitext(filepath) + (numb,))
            if path.exists(new_path):
                numb += 1
            else:
                return new_path
    return filepath


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    """
    Generate a cosine learning rate schedule with optional warmup.

    Arguments:
        base_value: The initial learning rate or weight decay value.
        final_value: The final learning rate or weight decay value.
        epochs: Total number of epochs.
        warmup_epochs: Number of warmup epochs. Default is 0.
        start_warmup_value: Starting value for warmup. Default is 0.

    Returns:
        A numpy array containing the learning rate or weight decay value for each epoch.
    """
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs)

    epoch_counter = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(epoch_counter * np.pi / epochs))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule


class EarlyStopper:
    """
    Early stopping utility to stop training when a certain criteria is met.

    Arguments:
        patience: Number of epochs to wait for improvement before stopping. Default is 5.
        min_delta: Minimum change to quantify as an improvement. Default is 0.
        baseline: Baseline value for comparison. Default is 0.

    Methods:
        early_stop(validation_loss): Determine if training should be stopped based on the validation loss.
    """
    def __init__(self, patience=5, min_delta=0, baseline=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.baseline = baseline
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if (validation_loss + self.min_delta) < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        if validation_loss < self.baseline:
            return True
        return False


def pretrain_model(model, pretraining_config, pretrain_method, dataset, dataset_test, save_weights, save_dir):
    """
    Pretrain a given model using the specified method.

    Arguments:
        model: The model to be pretrained.
        pretraining_config: Configuration parameters for pretraining.
        pretrain_method: The method to use for pretraining ('reconstruction' or other methods).
        dataset: The training dataset.
        dataset_test: The test dataset.
        save_weights: Boolean indicating whether to save the model weights.
        save_dir: Directory to save the model weights.

    Returns:
        The pretrained model, after applying the pretraining method.
    """
    if pretrain_method == "reconstruction":
        if pretraining_config.EARLY_STOPPING:
            early_stopper = EarlyStopper(patience=pretraining_config.PATIENCE,
                                         min_delta=pretraining_config.MIN_DELTA,
                                         baseline=pretraining_config.BASELINE)

        if pretraining_config.WITH_WARMUP:
            lr_schedule = cosine_scheduler(pretraining_config.LEARNING_RATE,
                                           pretraining_config.LR_FINAL,
                                           pretraining_config.NUM_EPOCHS,
                                           warmup_epochs=pretraining_config.LR_WARMUP,
                                           start_warmup_value=0)
        else:
            lr_schedule = cosine_scheduler(pretraining_config.LEARNING_RATE,
                                           pretraining_config.LR_FINAL,
                                           pretraining_config.NUM_EPOCHS,
                                           warmup_epochs=0,
                                           start_warmup_value=0)
        wd_schedule = cosine_scheduler(pretraining_config.WEIGHT_DECAY,
                                       pretraining_config.WD_FINAL,
                                       pretraining_config.NUM_EPOCHS,
                                       warmup_epochs=0,
                                       start_warmup_value=0)
        if pretraining_config.WITH_WD:
            optimizer = tfa.optimizers.AdamW(weight_decay=wd_schedule[0], learning_rate=lr_schedule[0])
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule[0])

        loss_lst = []
        test_loss_lst = []
        mse = tf.keras.losses.MeanSquaredError()
        initializer = True
        for epoch in range(pretraining_config.NUM_EPOCHS):

            optimizer.learning_rate = lr_schedule[epoch]
            optimizer.weight_decay = wd_schedule[epoch]

            for step, batch in enumerate(dataset):

                if initializer:
                    _ = model(batch[0])
                    print(model.Encoder.summary())
                    print(model.Decoder.summary())
                    print(model.summary())
                    initializer = False

                with tf.GradientTape() as tape:
                    [_, output] = model(batch[0])

                    loss = mse(batch[0], output)
                    loss_lst.append(loss)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # test loss
            for step, batch in enumerate(dataset_test):
                batch_t = batch[0]
                [_, output] = model(batch_t)
                test_loss = mse(batch_t, output)
                test_loss_lst.append(test_loss)

            wandb.log({
                "Train Loss": np.mean(loss_lst[-dataset.cardinality().numpy():]),
                "Valid Loss": np.mean(test_loss_lst[-dataset_test.cardinality().numpy():])})

            print("Epoch: ", epoch + 1, ", Train loss: ", np.mean(loss_lst[-dataset.cardinality().numpy():]),
                  ", Test loss: ", np.mean(test_loss_lst[-dataset_test.cardinality().numpy():]))
            if pretraining_config.EARLY_STOPPING:
                if early_stopper.early_stop(np.mean(test_loss_lst[-dataset_test.cardinality().numpy():])):  # test_loss
                    break

        if save_weights:  # add numbering system if file already exists
            save_dir = check_filepath_naming(save_dir)
            wandb.log({"Actual save name": save_dir})
            model.save_weights(save_dir)
