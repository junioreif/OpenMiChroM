from setuptools import setup, find_packages
from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.rst")) as f:
    long_description = f.read()

__version__ = "1.1.1"
for line in open(path.join("OpenMiChroM", "__init__.py")):
    if line.startswith("__version__"):
        exec(line.strip())

setup(
    name="OpenMiChroM",
    version=__version__,
    description="Open-Michrom lib for chromosome simulations",
    license="MIT",
    url="https://ndb.rice.edu/Open-MiChroM",
    author="Antonio Bento de Oliveira Junior,Vinicius de Godoi Contessoto",
    author_email="antonio.oliveira@rice.edu,contessoto@rice.edu",
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
    ],
    include_package_data=False,
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={
        "OpenMiChroM": ["share/MiChroM.ff"],
        "OpenMiChroM._cndb_stream": ["py.typed"],
        "OpenMiChroM._cndb_stream._vendor": ["README.md"],
        "OpenMiChroM._cndb_stream._vendor.hdf5_indexed_reader": ["LICENSE"],
    },
    python_requires=">=3.10",
    install_requires=['openmm', 'numpy', 'scipy', 'scikit-learn', 'h5py', 'pandas'],
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/x-rst",
)
