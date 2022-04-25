MN-BaB <img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg">
======== 
Multi-Neuron Guided Branch-and-Bound ([MN-BaB](https://www.sri.inf.ethz.ch/publications/ferrari2022complete)) is a state-of-the-art complete neural network verifier that builds on the tight multi-neuron 
constraints proposed in [PRIMA](https://www.sri.inf.ethz.ch/publications/mueller2021precise) and leverages these constraints within a BaB framework to yield an efficient, GPU based dual solver.
MN-BaB is developed at the [SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch/) as part of the [Safe AI project](http://safeai.ethz.ch/).


Installation
----------------------
<details>
  <summary>Instructions</summary>
  


     
#### Prerequisites


```
Linux:
sudo apt-get install m4
Mac:
brew install m4
```

```
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
sudo make install
cd ..
rm gmp-6.1.2.tar.xz


wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure
make
sudo make install
cd ..
rm mpfr-4.1.0.tar.xz

  
wget https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
tar zxf cddlib-0.94m.tar.gz
rm cddlib-0.94m.tar.gz
cd cddlib-0.94m
./configure
make
sudo make install
cd ..
```

#### ELINA setup
Go into top level directory of repo:
```
cd mn-bab
```
Setup ELINA:
```
git clone https://github.com/eth-sri/ELINA.git
cd ELINA
./configure -use-deeppoly -use-fconv
make
sudo make install
cd ..
```
#### Python Environment setup
```
python -m venv mn-bab-env
source mn-bab-env/bin/activate
  
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$PWD
```
</details>

Example usage
----------------------

```
python src/verify.py -c configs/cifar10_conv_small.json
```


Contributors
----------------------
* [Claudio Ferrari ](https://github.com/ferraric) - c101@gmx.ch
* [Mark Niklas Müller](https://www.sri.inf.ethz.ch/people/mark) - mark.mueller@inf.ethz.ch  
* [Nikola Jovanovic](https://www.sri.inf.ethz.ch/people/nikola) - nikola.jovanovic@inf.ethz.ch 

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