# Restoring Force Surface Models for Finite Element Modeling in Crash Analysis

Source code associated with a Master thesis completed as part of the requirements for an MSc in Computational Science and Engineering at the Technical University of Munich

## Source code overview
The version of python used is 3.9.10 but recent version of python >=3.7 is expected to be compatible. Required libraries can be installed using the provided (requirements.txt)[requirements.txt] file.

```bash
$ pip install -r requirements.txt
```

### Modules
Several modules contain convenience functions used in the investigation, specifically: 

* [evaluate.py](evaluate.py) - Contains the main modular evaluation function, which reads . The data is provided via the `mat` parameter (an instane of the `Example2D` class), specific models are provided via the `model` parameter (`scikit-learn` and `tensorflow.keras` models are compatible), and suitable training, testing and acceleration prediction functions via `trainfunc`, `testfunc` and `ar_predfunc` respectively. `pod_basis_size` contains the size of the basis for reduced displacement  and `eps_basis_size` for reduced strain (if applicable).

* [utils.py](utils.py) - Contains multiple helper classes and functions, the more important of which include:
    - `Example2D` is the main dataset class, handling input parsing from LS-DYNA as well as MOR transformations
    - The `step` and `evolve` functions implement central difference time integration scheme for a custom acceleration prediction funtion
    - The `roughplot2D` and `animatedplot2D` are used to create consitent plotting in a format suitable to the data
    - Other functions and classes are used for convenience or additional non-critical functionality

* [models.py](models.py) - Contains implementation of custom machine learning models


### Investigation notebooks
Four types of models have been investigated, each of them grouped in a separate `jupyter` notebook. The notebooks are mutually independent and contain results of multiple numerical experiments organized by topic, as well as defintions associated with defining the different types of models, the corresponding training procedures and prediction functions, as well as custom results presentation when required. Using `jupyter` outline functionality (via an [extension](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/toc2/README.html) or [VS Code](https://code.visualstudio.com/)) is strongly recommended for easy navigation.

* History-free linear models for both the hyperelastic and the plastic example are investigated in [investigate_plate_linear.ipynb](investigate_plate_linear.ipynb) (all other investigations use only data from the plastic example)
* Hysteretic linear models based either or time-delay or a physical history variable are investigated in [investigate_plate_hystlinear.ipynb](investigate_plate_hystlinear.ipynb)
* Linear-tree based explainable ensemble models are investigated in [investigate_plate_tree.ipynb](investigate_plate_tree.ipynb)
* Dense NN, RNN and LSTM deep learning models are investigated in [investigate_plate_nn.ipynb](investigate_plate_nn.ipynb)

### Data
All data used is generated using the [plate/plate_1_20_20.k](plate/plate_1_20_20.k) (hyperelastic example) and [plate/plate_2_20_20.k](plate/plate_2_20_20.k) (plastic example) keyfiles executed with LS-DYNA 12.0.0.

## Thesis Abstract 

Crash analysis is carried out in multiple industries to ensure mechanical structures meet a wide spectrum of safety requirements. Optimization, robustness studies, uncertainty quantification and other many-query applications require multiple simulation evaluations which would have a prohibitive computational cost for full scale explicit Finite Element (FE) models. The need to speed up simulation times and the opportunity to make use of existing data has driven the development of model order reduction (MOR) in the field.

One approach to efficient data-driven modeling respecting the constraint of non-intrusive MOR, which does not require modification of the code of established commercial solvers, is to quickly evaluate the non-linear contribution of internal forces and carry out the integration in reduced space. Rather than accomplishing this via hyper-reduction as has been previously established, this thesis explores surface models for the restoring force (also called force-state mapping models). Such models approximate the internal forces in the reduced space from the available reduced state and have been successful in approximating non-linear effects in hyperelastic materials. Adapting these models to hysteretic phenomena such as plasticity has the potential to combine the time extrapolation capability of intrusive methods with the independent implementation of non-intrusive techniques and the associated access to powerful data-driven machine learning tools for the purposes of crash analysis.

Four types of models are evaluated in this thesis for their potential in being able to reproduce the evolution of a simple example in reduced space. Existing history-free linear models are shown incapable of dealing with plastic effects. Further the advantages and limitations of novel hysteretic models, based on time delay or physics based history variables such as effective plastic strain, are assessed. Two complex model types, based on a linear tree ensemble and several types of neural networks are also explored but with limited success. Multiple challenges are identified which prevent realizing the full potential
of explicitly time integrated fully non-intrusive surrogates of FE models in crash analysis. Nevertheless, in addition to some approaches where restoring force surface modeling comes short, several promising directions for development are also identified.