# Cultural Change Among r/lgbt Subreddit Users, 2008-2018
### Dynamic Topic Modeling using BERTopic
Author: Devon Rojas
Last Updated: December 12, 2024

Requires python 3.12\
Use provided environment.yml file to install conda environment used for this project

#### Replicating Analysis 
Analyses presented in paper can be run directly using the analyses.ipynb file, but files for entire analysis pipeline are also included.

#### To run from beginning
The data/raw folder contains raw data of r/lgbt extracted from ConvoKit's Pushshift.io Reddit repository. First, run 0_prep_data.py to generate the combined data frame used to train the BERTopic model. Then, run 1_make_model.py to generate the model specified in the paper. NOTE: Because documents must be embedded, the training process may take a long time depending on computer architecture/resource availability. I ended up running this pipeline on a computing cluster with 4 cores and a minimum of 16GB of memory.
