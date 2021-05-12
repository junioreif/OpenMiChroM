from setuptools import setup, find_packages
from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md")) as f:
    long_description = f.read()

__version__ = "0.0.1"
for line in open(path.join("OpenMiChroM", "__init__.py")):
    if line.startswith("__version__"):
        exec(line.strip())

setup(
    name="OpenMiChroM",
    version=__version__,
    description="Open-Michrom lib for chromosome simulations",
    url="https://ndb.rice.edu/Open-MiChroM",
    author="Antonio Bento de Oliveira Junior / Vinicius de Godoy Contessoto",
    author_email="antonio.oliveira@rice.edu",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.4',
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
    ],
    packages=find_packages(),
    install_requires=['numpy>1.18', 'scipy', 'sklearn', 'six', 'h5py>2', 'openmm>1.6'],
    entry_points={"console_scripts": ["CLINAME=OpenMiChroM._cli:main"]},
    zip_safe=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
