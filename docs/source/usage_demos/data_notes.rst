A Note About HDF5 Data
======================

Most of the data input-output is done using the Hierarchical Data Format `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ file structure, interfaced through H5PY (``h5py`` package) and PyTables (``tables`` package).

Advantages of HDF5

* Flexible tree structure: can store/remove almost any data objects in a path like hierarchy
* Optimized for very large "flat" data objects (e.g. array timeseries)

  - B-Tree lookups can scan to arbitrary data points (*very fast* compared to "seek" operations on binary files)

* Widely adopted convention for data storage (e.g. modern MATLAB ".mat" files)
    
The file extension convention used here is "``*.h5``\ ", but of course "``*.mat``\" files also work if they are HDF5.

Generic data stashing & loading
-------------------------------

The hierarchical structure of .h5 files are ideal to wrap the generic :py:class:`Bunch <ecogdata.util.Bunch>` objects.
A "Bunch" is the conventional name for a simple container object that is a dictionary, but acts like a "dot-accessible" container.
They are a bit like the ``struct`` in MATLAB, and I use them as such in place of more feature-full objects.

Simple Bunch storage
~~~~~~~~~~~~~~~~~~~~

>>> from ecogdata.util import Bunch                                                                                                                                                                            
>>> from ecogdata.datastore import load_bunch, save_bunch                                                                                                                                                      
>>> b1 = Bunch()                                                                                                                                                                                               
>>> b1.a_list = [1, 2, 3]                                                                                                                                                                                      
>>> b1.a_string = 'asdf'
>>> b1                                                                                                                                                                                                         
a_list   : <class 'list'>
a_string : <class 'str'>
>>> save_bunch('stashed_bunch_single.h5', '/', b1)                                                                                                                                                             
>>> load_bunch('stashed_bunch_single.h5', '/')                                                                                                                                                                 
a_list   : <class 'list'>
a_string : <class 'str'>
>>> load_bunch('stashed_bunch_single.h5', '/') == b1                                                                                                                                                           
True

Meanwhile the HDF5 file has this structure.
Note some data types are "pickled" rather than stored natively.
(Every data store file has a "b_pickle" entry, which is usually empty.)

::

   $ h5dump --contents stashed_bunch_single.h5 
   HDF5 "stashed_bunch_single.h5" {
   FILE_CONTENTS {
    group      /
    dataset    /a_list
    dataset    /b_pickle
    }
   }

Nested Bunch
~~~~~~~~~~~~

Bunches can be tree-like (any attribute can also be a Bunch), and they can also be stored hierarchically.

Storing one hierarchical bunch:

>>> b2 = Bunch()                                                                                                                                                                                               
>>> b2.a_list = [1, 2, 3]                                                                                                                                                                                      
>>> b2.sub_bunch = Bunch(another_list=[3, 2, 1])                                                                                                                                                               
>>> save_bunch('stashed_hierarchical_bunch.h5', '/', b2)

This HDF5 has a sub-group named "/sub_bunch"

::

   $ h5dump --contents stashed_hierarchical_bunch.h5 
    HDF5 "stashed_hierarchical_bunch.h5" {
    FILE_CONTENTS {
    group      /
    dataset    /a_list
    dataset    /b_pickle
    group      /sub_bunch
    dataset    /sub_bunch/another_list
    dataset    /sub_bunch/b_pickle
    }
   }

Now these Bunches can be loaded entirely, as a nested Bunch, or separately. For example:

>>> load_bunch('stashed_hierarchical_bunch.h5', '/sub_bunch') == b2.sub_bunch                                                                                                                                  
True
>>> load_bunch('stashed_hierarchical_bunch.h5', '/') == b2                                                                                                                                                     
True

Data stores: multiple Bunches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, multiple Bunches can be stored to different "leaves" within in the same HDF5 file.

.. attention::
   This is very helpful for collecting preprocessed or intermediate results across a number of separate different trials/experiments!

>>> save_bunch('stashed_bunches.h5', '/b1', b1)                                                                                                                                                                
>>> save_bunch('stashed_bunches.h5', '/b2', b2)  

Now the file contents indicate two top-level groups ("b1" and "b2"), and the "b2" group also has a sub-group "sub_bunch".
As shown before, any of these Bunches can be retreived separately (along with any Bunches nested within their level).

::

   $ h5dump --contents stashed_bunches.h5 
    HDF5 "stashed_bunches.h5" {
    FILE_CONTENTS {
    group      /
    group      /b1
    dataset    /b1/a_list
    dataset    /b1/b_pickle
    group      /b2
    dataset    /b2/a_list
    dataset    /b2/b_pickle
    group      /b2/sub_bunch
    dataset    /b2/sub_bunch/another_list
    dataset    /b2/sub_bunch/b_pickle
    }
   }

Extra functionality
~~~~~~~~~~~~~~~~~~~

The :py:func:`ecogdata.datastore.h5utils.save_bunch` and :py:func:`ecogdata.datastore.h5utils.load_bunch` have advanced options to consider.
For permanent data stores, be careful with the "mode" parameter of "save_bunch".
The mode is "a" by default, which will add to existing files or create one if needed.
However, any call with ``mode='w'`` will wipe out an existing file.
Sometimes it is necessary to over-write certain paths in an existing file.
For example, you may correct a calculation affecting only one data object.
You can use "overwrite_paths=True" in this case:

>>> del b2.sub_bunch                                                                                                                                                                                           
>>> save_bunch('stashed_bunches.h5', '/b2', b2, overwrite_paths=True)

The result does not affect the rest of the file, but only the /b2 path

::

   $ h5dump --contents stashed_bunches.h5 
    HDF5 "stashed_bunches.h5" {
    FILE_CONTENTS {
    group      /
    group      /b1
    dataset    /b1/a_list
    dataset    /b1/b_pickle
    group      /b2
    dataset    /b2/a_list
    dataset    /b2/b_pickle
    }
   }
