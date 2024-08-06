from setuptools import setup, find_packages
from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.rst")) as f:
    long_description = f.read()

__version__ = "1.0.9"
for line in open(path.join("OpenMiChroM", "__init__.py")):
    if line.startswith("__version__"):
        exec(line.strip())

setup(
    name="OpenMiChroM",
    version=__version__,
    description="Open-Michrom lib for chromosome simulations",
    url="https://ndb.rice.edu/Open-MiChroM",
    author="Antonio Bento de Oliveira Junior,Vinicius de Godoi Contessoto",
    author_email="antonio.oliveira@rice.edu,contessoto@rice.edu",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
    ],
    include_package_data=True,
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn', 'six', 'h5py', 'pandas', 'seaborn'],
    entry_points={"console_scripts": ["CLINAME=OpenMiChroM._cli:main"]},
    zip_safe=True,
    long_description=long_description,
    long_description_content_type="text/x-rst",
)
