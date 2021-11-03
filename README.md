# ecogdata: data loading for Viventi lab recordings

![example workflow](https://github.com/miketrumpis/ecogdata/actions/workflows/build_wheels.yml/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/miketrumpis/ecogdata/branch/master/graph/badge.svg?token=H1ROCJZPC7)](https://codecov.io/gh/miketrumpis/ecogdata)

This library provides data preprocessing and packaging for recording systems used by the Viventi lab for micro-electrocorticography. 

* Data wrangling from numerous data acquisition systems:

  + Open-ephys (.continuous files)
  + National instruments (.tdms files)
  + Intan (.rhd files, but needs HDF5 conversion: https://github.com/miketrumpis/rhd-to-hdf5)
  
* Parallelized array signal processing with shared memory
* HDF5-based memory mapping and dataset storage
* Array read/write abstraction spanning multiple mapped input files
* Trial/stimulus event alignment

## Install

Preliminary: set up your virtualenv or conda env. 
With conda, you can use the provided YAML file to create a working environment. 

```bash
$ conda env create --file conda_env.yml --name <your-env-name>
```

Then clone ecogdata:

```bash
$ git clone https://github.com/miketrumpis/ecogdata.git
```

Last, use pip to install ecogdata in any way you choose. 
I use "editable" mode to avoid re-installing after git pulls: pip install -e 

```bash
$ python -m pip install -e ./ecogdata
```

Note that pip needs to differentiate installing from a path versus from PyPI, hence the contrived "./" path syntax.
If this is not working in a non-unix command terminal, then do this instead (and likewise for the following instructions):

```bash
$ cd ecogdata
$ python -m pip install -e .
```

To run tests on the source code, install a variation package and run ``pytest``.
(If using conda, you may want to install pytest through conda.)

```bash
$ python -m pip install -e ./ecogdata[test]
$ python -m pytest ecogdata/ecogdata
```

## Docs & demo notebooks

To build API documentation and usage demos, first install docs requirements ([docs] package variation) and run ``sphinx``.
(If using conda, you may want to install the docs requirements in ``setup.cfg`` through conda.)

```bash
$ python -m pip install -e ./ecogdata[docs]
$ cd ecogdata/docs
$ make all
```

Alternatively, install ``jupyter`` and run the notebooks in ``docs/source/usage_demos`` interactively.