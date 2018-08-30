=====================
How to install CumuPy
=====================

---------------------
Pre-requisites
---------------------
As mentioned in `<README.rst>`_, in order to run (and even install) CumuPy you need a certain number of libraries

- NumPy and SciPy (http://www.numpy.org/, called ``python-numpy python-scipy`` on debian/ubuntu)
- Matplotlib (https://matplotlib.org/ called ``python-matplotlib`` on ubuntu/debian)
- pyFFTW (https://pypi.python.org/pypi/pyFFTW)

  This is only a python wrapper to the FFTW3 library, so be sure to have this installed, together with its development version (for example on ubuntu/debian install both ``libfftw3`` and ``libfftw3-dev``.

---------------------
Installation
---------------------

In order to use CumuPy in your computer, you have several possibilities:

1. You can install the entire project onto your pc, using the pip command. Fetch the wheel CumuPy package `here <https://github.com/mxz2013/CumuPy/raw/master/dist/CumuPy-1.0.0-py2-none-any.whl>`_ and give
   
   ``pip install CumuPy-1.0.0-py2-none-any.whl`` 

   This command will also install for you ``numpy``, ``scypy``, ``matplotlib`` and ``pyfftw`` with the correct version. 
   Be sure to have at least ``libfftw3`` and ``libfftw3-dev`` installed on your system.
   
   Normally, this would install the package (and the tests) into ``/home/user/.local/lib/python2.7/site-packages/CumuPy``.

2. You can clone it (you do not need an account on github to do so)

   ``git clone https://github.com/mxz2013/CumuPy.git``

   and start to use the code immediately (the main module is ``CumuPy/cumupy.py``)

3. You can download the code from `here <https://github.com/mxz2013/CumuPy/archive/master.zip>`_

   unzip it, and use  the code immediately (the main module is ``CumuPy/cumupy.py``)

