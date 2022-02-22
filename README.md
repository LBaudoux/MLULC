# Multi Land-use/Land-cover Translation Network
 This repo is still subject to change. In paricular, we will improve the README installation procedure, the code comments and the influence of each parameter.
 This github repository follow the structure proposed by [Hager Rady](https://github.com/hagerrady13/) and [Mo'men AbdelRazek](https://github.com/moemen95). For more information on the repository structure, and how to handle launch of the code refers to this [github repo](https://github.com/moemen95/Pytorch-Project-Template).

## Installation

1. Clone the repository
2. Inside the data folder download and unzip the dataset  [zenodo archive](https://doi.org/10.5281/zenodo.5843595). Read the dataset README for more information.
3. Either Use anaconda to install required python module or replicate the environment used for our experimentation using the environement.yml file provided (note that this environment holds some unnecessary module )

## launch

- default parameters used for our experimentation are provided in the config folder
- to launch an experiment use the run.sh file after choosing the desired config file.

#Tips for Custom dataset.

If you want to use a custom dataset you need  : 
1. To crop your maps in tiles with a reasonable width (big tiles wont fit in memory, small one will give few spatial context information ).
2. Either store them in an hdf5 with the same attribute as those describe in the [zenodo archive](https://doi.org/10.5281/zenodo.5843595) README or adapt the datasets/landcover_to_landcover.py file to read your folder.
### Acknowledgement
* The French National Research agency as a part of the MAESTRIA project (grant ANR-18-CE23-0023) for funding.
* The AI4GEO project (http://www.ai4geo.eu/) for material support.
* [Hager Rady](https://github.com/hagerrady13/) and [Mo'men AbdelRazek](https://github.com/moemen95) for the repository template


### License:
This project is licensed under MIT License - see the LICENSE file for details