from setuptools import setup, find_packages

VERSION = '0.0.1'
DISTNAME = 'esio'
PACKAGES = ['esio']
DESCRIPTION = "Extended Sea Ice Outlook"
AUTHOR = 'Nic Wayand'
AUTHOR_EMAIL = 'nicway@gmail.com'
URL = 'https://github.com/NicWayand/ESIO'
LICENSE = 'GPL-3.0'
PYTHON_REQUIRES = '>=3.5'
INSTALL_REQUIRES = ['xarray', 'numpy', 'scipy', 'cartopy', 'seaborn','pytest', 'matplotlib']
CLASSIFIERS = [
    'Development Status :: 1 - Beta',
    'License :: OSI Approved :: GPL-3.0 License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
]


def readme():
    with open('README.md') as f:
        return f.read()


setup(name=DISTNAME,
      version=VERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=readme(),
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      url=URL,
      packages=PACKAGES)
