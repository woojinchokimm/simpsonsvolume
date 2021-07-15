## Overview

**simpsonsvolume** is a toolbox used for volumetric analysis of 3D anatomical meshes of the left ventricle using different interpretations of the **Simpson's bi-plane rule**. The meshes are reconstructed a retrospective cohort representative of:

* an adult population [UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/)
* a young population (preterm) 
* an elderly heart failure population.

**Note** This repository only contains the code, not the imaging data. The meshes have been made publicly available [here](dx.doi.org/10.6084/m9.figshare.14933463).

## Installation

The toolbox is developed using [Python](https://www.python.org) programming language. The toolbox is developed specifically using Python 3.7.

The toolbox depends on some external libraries which need to be installed, including:

* vtk for mesh manipulation;
* numpy and scipy for numerical computation;
* matplotlib for data visulisation;
* pandas for handling spreadsheet;
* vg for geometrical calculations;

The most convenient way to install these libraries is to use pip3 (or pip for Python 2) by running this command in the terminal:
```
pip3 install numpy scipy matplotlib pandas vtk vg
```

## Usage

**Data preparation** There is a directory named *data*, which contains the scripts for preparing the training dataset. For a machine learning project, data preparation step including acquisition, cleaning, format conversion etc normally takes at least the same amount of your time and headache, if nor more, as the machine learning step. But this is a crucial part, as all the following work (your novel machine learning ideas) needs the data.

```
root
│   README.md  
│
└───data_handlers
│   │   data_viz.py
│   │   heart.py
│   │   rv_dir_.py
│   │   vtkfunctions.py
│   │   vtkslicer.py
│   
└───meshes
│   │
│   └───UKBB_meshes
│       │   Case1.vtk
│       │   Case2.vtk
│       │   ...
│       │   anomaly.txt
│       │   apex_manual.txt
│       │   apex_nodes.txt
 
│   └───YHC_meshes
│       │   ...
   
│   └───HFC_meshes
│       │   ...
│    
└───output
│   └───UKBB_meshes (population 1)
│       └───BOD (method 1)
│          └───90
|              |  Case1_data_dict.pkl
|              |  Case2_data_dict.pkl
|              | ...

│          └───60
|              | ...  

│       └───TBC (method 2)
|           | ...

│       └───SBR (method 3)
|           | ...

│   └───YHC_meshes (population 2)
|           | ...

│   └───HFC_meshes (population 3)
|           | ...
```





