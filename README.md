# FVAB Quantum Project
Benchmark Quantum–Classical su Embedding OCT

## About
Repo for the FVAB Quantum Project.
Quantum computing is a new technology that is slowly entering our daily lives, not in the ways we expect though.
The machine learning approach, that's been stagnat for a while now, is finding new life with the ignition of the Quantum spark.
This project focuses on benchmarking the differences between the Quantum Machine Learning approach and the Classical Machine Learning approach by comparing them using a fixed pipeline.
## Needed
What's needed for the pipeline (It is possible to run the environment script to download and install all the dependencies) :
- Python
- PyTorch
- Jupyter

## Setup
1) Run the feature_extractor.ipynb
2) Run the PCASweep.ipynb
3) Each Ansatz is labeled in a specific way (C1, C2, C3), feel free to run any (1: RealAmplitudes, 2: EfficientSU2, 3: Ladder)

## Structure
Each folder has a specific job, in particular we have:
- artifact: Parent folder to store all the artifacts
 - metrics: Folder to store the JSON output files
 - circuit: Folder to store the images of the circuits
 - prediction: Folder to store the angles at the best checkpoint
 - weights: Folder to store the npz files
- Compressed Features: Folder to store the mid outputs of every phase and on each seed
- Dataset: Folder to store the labels of the dataset
- Extracted Features: Folder to store the features extracted from the ImageNet
- src: Source Code folder for each script and ipynb 
