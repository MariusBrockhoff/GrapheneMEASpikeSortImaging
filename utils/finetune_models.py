import keras.backend as K
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from utils.pretrain_models import *
import wandb

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.

    The function computes the accuracy of the clustering by finding an optimal match
    between the cluster labels and the true labels using the Hungarian algorithm.

    Arguments:
        y_true: True labels, numpy.array with shape `(n_samples,)`.
        y_pred: Predicted labels, numpy.array with shape `(n_samples,)`.

    Returns:
        accuracy: A float value between 0 and 1 indicating the clustering accuracy.
    """
    # y_pred = y_pred.argmax(1)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def acc_tf(y_true, y_pred):
    """
    Calculate clustering accuracy for TensorFlow tensors.

    Similar to `acc` but specifically designed for use with TensorFlow tensors.

    Arguments:
        y_true: True labels, numpy.array with shape `(n_samples,)`.
        y_pred: Predicted labels, numpy.array with shape `(n_samples,)`.

    Returns:
        accuracy: A float value between 0 and 1 indicating the clustering accuracy.
    """
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_pred = y_pred.argmax(1)
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


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
            newPath = "{0}_{2}{1}".format(*path.splitext(filepath) + (numb,))
            if path.exists(newPath):
                numb += 1
            else:
                return newPath
    return filepath


def flatten_list(nested_list):
    """
    Flatten a nested list.

    Arguments:
        nested_list: A list possibly containing nested lists.

    Returns:
        A single flattened list.
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        elif isinstance(item, (int, float)):  # Check for integers and floats
            flattened.append(item)
    return flattened


class ClusteringLayer(tf.keras.layers.Layer):
    """
    Clustering layer used in deep embedded clustering models.

    This layer converts input samples (features) to soft labels, which are vectors representing
    the probability of the sample belonging to each cluster. The probability is calculated
    with a Student's t-distribution.

    Arguments:
        n_clusters: Number of clusters.
        weights: Initial cluster centers. Default is None.
        alpha: Degrees of freedom parameter in Student's t-distribution. Default is 1.0.

    Input shape:
        2D tensor with shape `(n_samples, n_features)`.

    Output shape:
        2D tensor with shape `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        self.built = None
        self.clusters = None
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = tf.keras.layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),
                                        initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def target_distribution(q):
    """
    computing an auxiliary target distribution
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


class BaseModelDEC(tf.keras.Model):
    """
    Base model for Deep Embedded Clustering (DEC).

    Arguments:
        encoder: Encoder model to transform input data.
        n_clusters: Number of clusters.

    The model combines an encoder with a clustering layer.
    """
    def __init__(self, encoder, n_clusters):
        super(BaseModelDEC, self).__init__()

        self.Encoder = encoder
        self.n_clusters = n_clusters
        self.clustering = ClusteringLayer(self.n_clusters, name='clustering')

    def call(self, inputs):
        logits = self.Encoder(inputs)
        out = self.clustering(logits)
        return out


class BaseModelIDEC(tf.keras.Model):
    """
    Base model for Improved Deep Embedded Clustering (IDEC).

    Arguments:
        encoder: Encoder model to transform input data.
        decoder: Decoder model for reconstructing input data from encoded representation.
        n_clusters: Number of clusters.

    The model combines an encoder, a decoder, and a clustering layer for clustering and reconstruction tasks.
    """
    def __init__(self, encoder, decoder, n_clusters):
        super(BaseModelIDEC, self).__init__()

        self.Encoder = encoder
        self.Decoder = decoder
        self.n_clusters = n_clusters
        self.clustering = ClusteringLayer(self.n_clusters, name='clustering')

    def call(self, inputs):
        logits = self.Encoder(inputs)
        out = self.Decoder(logits)
        clus = self.clustering(logits)
        return clus, out


class IDEC(object):
    """
    IDEC (Improved Deep Embedded Clustering) class.

    This class implements the IDEC algorithm, which enhances DEC with a reconstruction loss.

    Arguments:
        model: A pre-trained autoencoder model.
        input_dim: Dimensionality of the input data.
        n_clusters: Number of clusters.
        alpha: Parameter in Student's t-distribution. Default is 1.0.
        batch_size: Batch size for training. Default is 256.

    Methods include initialization, weight loading, feature extraction, cluster prediction, and the clustering process.
    """
    def __init__(self,
                 model,
                 input_dim,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(IDEC, self).__init__()

        self.autoencoder = model
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.model = None

    def initialize_model(self, ae_weights=None, gamma=0.1, optimizer='adam'):
        if ae_weights is not None:  # load pretrained weights of autoencoder
            dummy = tf.zeros(shape=[1, self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
            self.autoencoder(dummy)
            self.autoencoder.load_weights(ae_weights)
        else:
            print('ae_weights, i.e. path to weights of a pretrained model must be given')
            exit()

        # prepare IDEC model
        self.model = BaseModelIDEC(self.autoencoder.Encoder, self.autoencoder.Decoder, self.n_clusters)
        dummy = tf.zeros(shape=[1, self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
        _ = self.model(dummy)
        print(self.model.summary())

        # prepare IDEC model
        self.model.compile(loss={'output_1': 'kld', 'output_2': 'mse'},
                           loss_weights=[gamma, 1],
                           optimizer=optimizer)

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.model.Encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   save_IDEC_dir='./results/idec'):

        print('Update interval', update_interval)
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
        print('Save interval', save_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.model.Encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc_var = np.round(acc(y, y_pred), 5)
                    nmi_var = np.round(nmi(y, y_pred), 5)
                    ari_var = np.round(ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc_var, nmi_var, ari_var),
                          ' ; loss=', loss)
                    wandb.log({"Iteration": ite, "Accuracy": acc_var, "NMI": nmi_var, "ARI": ari_var, "Loss 1": loss[0],
                               "Loss 2": loss[1], "Loss 3": loss[2]})

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0] > index * self.batch_size:

                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=[p[index * self.batch_size::], x[index * self.batch_size::]])

                index = 0

            elif (index + 1) * self.batch_size > x.shape[0] and index * self.batch_size == x.shape[0]:
                index = 0

            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                    x[index * self.batch_size:(index + 1) * self.batch_size]])
                index += 1

            ite += 1

        save_IDEC_dir = check_filepath_naming(save_IDEC_dir)
        self.model.save_weights(save_IDEC_dir)
        return y_pred


class DEC(object):
    """
    DEC (Deep Embedded Clustering) class.

    This class implements the DEC algorithm, focusing solely on clustering without reconstruction.

    Arguments:
        model: A pre-trained autoencoder model.
        input_dim: Dimensionality of the input data.
        n_clusters: Number of clusters.
        alpha: Parameter in Student's t-distribution. Default is 1.0.
        batch_size: Batch size for training. Default is 256.

    Methods include initialization, weight loading, feature extraction, cluster prediction, and the clustering process.
    """
    def __init__(self,
                 model,
                 input_dim,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):

        super(DEC, self).__init__()

        self.autoencoder = model
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.model = None

    def initialize_model(self, optimizer, ae_weights=None):
        if ae_weights is not None:  # load pretrained weights of autoencoder
            dummy = tf.zeros(shape=[512, self.input_dim[0]], dtype=tf.dtypes.float32, name=None)
            self.autoencoder(dummy)
            self.autoencoder.load_weights(ae_weights)
        else:
            print('ae_weights, i.e. path to weights of a pretrained model must be given')
            exit()

        # prepare DEC model
        self.model = BaseModelDEC(self.autoencoder.Encoder, self.n_clusters)
        self.model.compile(loss='kld', optimizer=optimizer)
        self.model.build((None, self.input_dim[0]))
        print(self.model.summary())

    def load_weights(self, weights_path):  # load weights of DEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.model.Encoder.predict(x)

    def predict_clusters(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self, x, y=None,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   save_DEC_dir='./results/dec'):

        print('Update interval', update_interval)

        # initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.model.Encoder.predict(x))
        y_pred_last = y_pred
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        loss = 0
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc_var = np.round(acc(y, y_pred), 5)
                    nmi_var = np.round(nmi(y, y_pred), 5)
                    ari_var = np.round(ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc_var, nmi_var, ari_var),
                          ' ; loss=', loss)
                    wandb.log({"Iteration": ite, "Accuracy": acc_var, "NMI": nmi_var, "ARI": ari_var, "Loss": loss})

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0] > index * self.batch_size:

                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=p[index * self.batch_size::])
                index = 0

            elif (index + 1) * self.batch_size > x.shape[0] and index * self.batch_size == x.shape[0]:
                index = 0
                loss = 0

            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=p[index * self.batch_size:(index + 1) * self.batch_size])
                index += 1

            ite += 1

        save_DEC_dir = check_filepath_naming(save_DEC_dir)
        self.model.save_weights(save_DEC_dir)
        return y_pred


def finetune_model(model, finetune_config, finetune_method, dataset, dataset_test, benchmark=False):
    """
    Finetune a given model using DEC or IDEC method.

    Arguments:
        model: The model to be fine-tuned.
        finetune_config: Configuration parameters for fine-tuning.
        finetune_method: The method to use for fine-tuning ('DEC' or 'IDEC').
        dataset: The training dataset.
        dataset_test: The test dataset.
        benchmark: If run on simulated, benchmarking data. default: False

    Returns:
        y_pred_finetuned: The predicted labels after fine-tuning.
        y: The true labels.
    """
    # data prep --> combined train and test
    y = None
    for step, batch in enumerate(dataset):
        if step == 0:
            x = batch[0]
            if benchmark:
                y = batch[1]
        else:
            x = np.concatenate((x, batch[0]), axis=0)
            if benchmark:
                y = np.concatenate((y, batch[1]), axis=0)

    for stept, batcht in enumerate(dataset_test):
        x = np.concatenate((x, batcht[0]), axis=0)
        if benchmark:
            y = np.concatenate((y, batcht[1]), axis=0)

    if finetune_method == "DEC":

        dec = DEC(model=model,
                  input_dim=x[0, :].shape,
                  n_clusters=finetune_config.DEC_N_CLUSTERS,
                  batch_size=finetune_config.DEC_BATCH_SIZE)
        dec.initialize_model(
            optimizer=tf.keras.optimizers.SGD(learning_rate=finetune_config.DEC_LEARNING_RATE,
                                              momentum=finetune_config.DEC_MOMENTUM),
            ae_weights=finetune_config.PRETRAINED_SAVE_DIR)

        y_pred_finetuned = dec.clustering(x, y=y, tol=finetune_config.DEC_TOL,
                                              maxiter=finetune_config.DEC_MAXITER,
                                              update_interval=finetune_config.DEC_UPDATE_INTERVAL,
                                              save_DEC_dir=finetune_config.DEC_SAVE_DIR)


    elif finetune_method == "IDEC":

        idec = IDEC(model=model, input_dim=x[0, :].shape, n_clusters=finetune_config.IDEC_N_CLUSTERS,
                    batch_size=finetune_config.IDEC_BATCH_SIZE)
        idec.initialize_model(
            optimizer=tf.keras.optimizers.SGD(learning_rate=finetune_config.IDEC_LEARNING_RATE,
                                              momentum=finetune_config.IDEC_MOMENTUM),
            ae_weights=finetune_config.PRETRAINED_SAVE_DIR, gamma=finetune_config.IDEC_GAMMA)

        y_pred_finetuned = idec.clustering(x, y=y, tol=finetune_config.IDEC_TOL,
                                               maxiter=finetune_config.IDEC_MAXITER,
                                               update_interval=finetune_config.IDEC_UPDATE_INTERVAL,
                                               save_IDEC_dir=finetune_config.IDEC_SAVE_DIR)

    return y_pred_finetuned, y
