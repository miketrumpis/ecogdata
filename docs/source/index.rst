.. |date| date::
.. |time| date:: %H:%M

Welcome to ecogdata's documentation!
====================================

This library provides data preprocessing and packaging for recording systems used by the Viventi lab for micro-electrocorticography.

* Data wrangling from numerous data acquisition systems:

  + Open-ephys (.continuous files)
  + National instruments (.tdms files)
  + Intan (.rhd files)

* Parallelized array signal processing with shared memory
* HDF5-based memory mapping
* Array read/write abstraction spanning multiple mapped input files
* External timestamp alignment

Quick install
-------------

Generic steps for cloning ecogdata and installing follow.
If you are using `conda`_ or `pyenv`_ then activate environments and/or change the install procedure accordingly.

.. code-block:: bash

    $ git clone git@github.com:miketrumpis/ecogdata.git
    $ cd ecogdata
    $ pip install -r requirements.txt
    $ pip install .

More about ecogdata
-------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   usage_demos/index
   api
..   apidocs/modules

Misc
----

Documents rebuilt |date| at |time|.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _conda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _pyenv: https://github.com/pyenv/pyenv