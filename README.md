# Deep Embedding based Spike Sorting

## Prerequisites
You will need the following to run our code:
* Python 3
* Conda
* Weights and Biases (recommended, optional)

## Getting started
### Launch the virtual environment
1. We recommend launching a virtual environment to install all the necessary Python packages. Multiple options are available, when using conda, use:

    `conda create --name <env_name> --file requirements.txt`
2. To launch the created virtual environment that contains all the necessary Python packages, run
`conda activate env_name` from the root directory. 
The virtual environment should be active. Make sure the correct versions of CUDA and cuDNN are installed.

## Run DEC/IDEC spike sorting
### For Benchmarking on simulated spike recordings
For benchmarking of simulated spike shape recordings, spike sorting can be run in 'benchmark=True' mode.
Here, it's assumed that Ground truth labels for each spike are available. Input are Python .pkl files with the isolated spike shape recordings (see publication below for details on the structure).
The full spike sorting pipeline can be run on a sample dataset at path `sample_path` by calling:
`python run.py --Pretrain_Method reconstruction --Finetune_Method DEC/IDEC --Model DenseAutoencoder --PathData sample_path --Benchmark`.
Models and results are saved after pre-training. Therefore, pre-training or fine-tuning can also be called separately by running e.g.
`python run.py --Pretrain_Method reconstruction --Model DenseAutoencoder --PathData sample_path --Benchmark` or 
`python run.py --Finetune_Method DEC/IDEC --Model DenseAutoencoder --PathData sample_path --Benchmark`.

### For new, raw MEA recording data
Here, spike sorting shall be applied to classify all spikes from a raw MEA recording. Currently, we only support Multi Channel system acquisition systems (stored as .h5 files). The Raw recordings are filtered, spikes detected and pre-processed and subsequently all sorted at once (aim for high-level sorting).
For a sample raw recording file at path `sample_raw`, run:
`python run.py --Pretrain_Method reconstruction --Finetune_Method DEC/IDEC --Model DenseAutoencoder --PathData sample_raw`.

### Example Data
Example dataset for testing and use of this codebase is uploaded via Zenodo and can be found [here](https://doi.org/10.5281/zenodo.13351549).

### Important Notes
All parameters can be adjusted in the configuration files. There are separate files for pre-training and fine-tuning. The config_file for data_preprocessing is only relevant when not benchmarking. By default, all training is tracked with Weight and Biases. If you do not wish to track the training, add `--wand False` when running.


## Acknowledgements
If you use Deep Embedding based Spike Sorting, please cite our [publication]( https://doi.org/10.1002/advs.202402967):

Meng Lu, Ernestine Hui, Marius Brockhoff, Jakob Träuble, Ana Fernandez-Villegas, Oliver J Burton, Jacob Lamb, Edward Ward, Philippa J Hooper, Wadood Tadbier, Nino F Läubli, Stephan Hofmann, Clemens F Kaminski, Antonio Lombardo, Gabriele S Kaminski Schierle (2024). *Graphene Microelectrode Arrays, 4D Structured Illumination Microscopy, and a Machine Learning Spike Sorting Algorithm Permit the Analysis of Ultrastructural Neuronal Changes During Neuronal Signaling in a Model of Niemann–Pick Disease Type C*. Advanced Science, 2402967.
