# ecogdata: data loading for Viventi lab recordings

This library provides data preprocessing and packaging for recording systems used by the Viventi lab for micro-electrocorticography. 

* Data wrangling from numerous data acquisition systems:

  + Open-ephys (.continuous files)
  + National instruments (.tdms files)
  + Intan (.rhd files)
  
* Parallelized array signal processing with shared memory
* HDF5-based memory mapping
* Array read/write abstraction spanning multiple mapped input files
* External timestamp alignment
