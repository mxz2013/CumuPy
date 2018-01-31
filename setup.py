"""
A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Get the license from the LICENSE file
with open(path.join(here, 'LICENSE.txt'), encoding='utf-8') as f:
    license = f.read()
# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='CumuPy',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0.0',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='Spectral functions via the cumulant expansion approximation',  # Required

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/mxz2013/Cumulant_SPF',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='Jianqiang Sky Zhou， and Matteo Guzzo',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='Jianqiang.Zhou@polytechnique.edu',  # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.4',
        #'Programming Language :: Python :: 3.5',
        #'Programming Language :: Python :: 3.6',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='spectral function photoemission cumulant spectroscopy',  # Optional

    license = license,
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
#    py_modules = [
#	'CumuPy', 
#	'calc_qp',
#	'outread_modules',
#	'sf_modules_spin',
#	'sf_toc11',
#	'sf_gw',
#	'sf_rc',
#	],
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
	'numpy',
	'scipy',
	'matplotlib',
	'pyfftw>=0.10.4'
	],  # Optional

    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    #extras_require={  # Optional
    #    'dev': ['check-manifest'],
    #    'test': ['coverage'],
    #},

    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    #include_package_data=True,
    #packages=find_packages(exclude=('tools')),
    #packages=['CumuPy'],
    packages=find_packages(),
    #package_dir={'cumupy': 'CumuPy'},
    package_data={  # Optional
        '': ['*rst','*.dat'],
    },
    data_files=[
	('/CumuPy/test/Al', ['CumuPy/test/Al/wtk.dat']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/README.rst']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/sp.out']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/spo_DS3_SIG']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/cs_1100.0.dat']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/invar.in']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/make_input']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/pjt_d.dat']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/pjt_p.dat']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/pjt_s.dat']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/R_1100.0.dat']),
	('/CumuPy/test/Al', ['CumuPy/test/Al/wp_s.dat']),
	('/CumuPy/test/sodium_valence', ['CumuPy/test/sodium_valence/invar.in']),
	('/CumuPy/test/sodium_valence', ['CumuPy/test/sodium_valence/make_input']),
	('/CumuPy/test/sodium_valence', ['CumuPy/test/sodium_valence/README.rst']),
	('/CumuPy/test/sodium_valence', ['CumuPy/test/sodium_valence/spo_DS3_SIG']),
	('/CumuPy/test/sodium_valence', ['CumuPy/test/sodium_valence/sp.out']),
	('/CumuPy/test/sodium_valence', ['CumuPy/test/sodium_valence/wtk.dat']),
	('/CumuPy/test/sno2', ['CumuPy/test/sno2/hartree.dat']),
	('/CumuPy/test/sno2', ['CumuPy/test/sno2/invar.in']),
	('/CumuPy/test/sno2', ['CumuPy/test/sno2/out_gw_DS4_SIG']),
	('/CumuPy/test/sno2', ['CumuPy/test/sno2/README.srt']),
	('/CumuPy/test/sno2', ['CumuPy/test/sno2/sno2gw.out']),
	('/CumuPy/test/sno2', ['CumuPy/test/sno2/wtk.dat'])
	],  # Optional
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    #entry_points={  # Optional
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
)
