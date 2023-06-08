MN-BaB <img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg">
======== 
Multi-Neuron Guided Branch-and-Bound ([MN-BaB](https://www.sri.inf.ethz.ch/publications/ferrari2022complete)) is a state-of-the-art complete neural network verifier that builds on the tight multi-neuron 
constraints proposed in [PRIMA](https://www.sri.inf.ethz.ch/publications/mueller2021precise) and leverages these constraints within a BaB framework to yield an efficient, GPU based dual solver.
MN-BaB is developed at the [SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch/) as part of the [Safe AI project](http://safeai.ethz.ch/).

This version is an adaptation of the [VNN-COMP'22](https://arxiv.org/abs/2212.10376) entry allowing for the certification of models trained with the novel certified training method [SABR](https://openreview.net/forum?id=7oFuxtJtUMH), without modifications. 

### Cloning
This repository contains a submodule. Please make sure that you have access rights to the submodule repository for cloning. After that either clone recursively via 

```
git clone --branch SABR_ready --recurse-submodules https://github.com/eth-sri/mn-bab
```

or clone normally and initialize the submodule later on

```
git clone --branch SABR_ready https://github.com/eth-sri/mn-bab
git submodule init
git submodule update
```

There's no need for a further installation of the submodules.


### Installation
Create and activate a conda environment:

```
  conda create --name MNBAB python=3.7 -y
  conda activate MNBAB
  ```

This script installs a few necessary prerequisites including the ELINA library and GUROBI solver and sets some PATHS. It was tested on an AWS Deep Learning AMI (Ubuntu 18.04) instance.

```
source setup.sh
```

Install remaining dependencies:
```
python3 -m pip install -r requirements.txt
PYTHONPATH=$PYTHONPATH:$PWD
```

Download the full MNIST, CIFAR10, and TinyImageNet test datasets in the right format and copy them into the `test_data` directory:  
[MNIST](https://files.sri.inf.ethz.ch/sabr/mnist_test_full.csv)  
[CIFAR10](https://files.sri.inf.ethz.ch/sabr/cifar10_test_full.csv)  
[TinyImageNet](https://files.sri.inf.ethz.ch/sabr/tin_val.csv)  

### Example usage

```
python src/verify.py -c configs/cifar10_conv_small.json
```

Contributors
----------------------
* [Claudio Ferrari ](https://github.com/ferraric) - c101@gmx.ch
* [Mark Niklas Müller](https://www.sri.inf.ethz.ch/people/mark) - mark.mueller@inf.ethz.ch  
* [Nikola Jovanovic](https://www.sri.inf.ethz.ch/people/nikola) - nikola.jovanovic@inf.ethz.ch
* [Robin Staab]()
* [Dr. Timon Gehr](https://www.sri.inf.ethz.ch/people/timon)

Citing This Work
----------------------

If you find this work useful for your research, please cite it as:

```
@inproceedings{
    ferrari2022complete,
    title={Complete Verification via Multi-Neuron Relaxation Guided Branch-and-Bound},
    author={Claudio Ferrari and Mark Niklas Mueller and Nikola Jovanovi{\'c} and Martin Vechev},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=l_amHf1oaK}
}
```

License and Copyright
---------------------

* Copyright (c) 2022 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0)