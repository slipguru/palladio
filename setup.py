#!/usr/bin/python
# palladio setup script

# from distutils.core import setup
from setuptools import setup

# Package Version
# from palladio import __version__ as version
version='2.0.2rc3'

setup(
    name='palladio',
    version=version,

    description=('ParALleL frAmework for moDel selectIOn'),
    long_description=open('README.md').read(),
    author='Matteo Barbieri, Samuele Fiorini, Federico Tomasi',
    author_email='{matteo.barbieri, samuele.fiorini, federico.tomasi}@dibris.unige.it',
    maintainer='Matteo Barbieri, Samuele Fiorini, Federico Tomasi',
    maintainer_email='{matteo.barbieri, samuele.fiorini, federico.tomasi}@dibris.unige.it',
    url='https://github.com/slipguru/palladio',
    download_url = 'https://github.com/slipguru/palladio/tarball/'+version,
    classifiers=[
	'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license = 'FreeBSD',

    packages=['palladio', 'palladio.wrappers', 'palladio.config_templates'],
    install_requires=['numpy (>=1.10.1)',
              'scipy (>=0.16.1)',
              'scikit-learn (>=0.18)',
              'matplotlib (>=1.5.1)',
              'seaborn (>=0.7.0)',
              'joblib',
              # 'mpi4py (>=2.0.0)'
              ],
    scripts=['scripts/pd_run.py','scripts/pd_analysis.py'],
)
