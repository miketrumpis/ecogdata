# ecogdata: data loading for Viventi lab recordings

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
If using pip, install numpy (and scipy why not) manually before batch installing.
**Skip this if using conda.**

```bash
$ pip install numpy<=1.16.0
$ pip install scipy
```

Then clone and install ecogdata and dependencies:

```bash
$ git clone https://github.com/miketrumpis/ecogdata.git
```

Pip:

```bash
$ pip install -r ecogdata/requirements.txt
```

Conda: **change "tables" to "pytables" in requirements.txt** (and add conda forge channel to your settings to avoid "-c")

```bash
$ conda install -c conda-forge -n <your-env-name> --file requirements.txt
```

Last, install ecogdata in any way you choose. 
I use "editable" mode to avoid re-installing after git pulls: pip install -e 

```bash
$ pip install -e ./ecogdata
```

Run tests to check install:

```bash
$ nosetests ecogdata
```

## Docs & demo notebooks

To build API documentation and usage demos, first install requirements in requirements-docs.txt.
Then:

```bash
$ cd docs
$ make all
```

Alternatively, install ``jupyter`` and run the notebooks in ``docs/source/usage_demos`` interactively.