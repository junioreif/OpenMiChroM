from setuptools import setup, find_packages
from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md")) as f:
    long_description = f.read()

__version__ = "Undefined"
for line in open(path.join("PACKAGENAME", "__init__.py")):
    if line.startswith("__version__"):
        exec(line.strip())

setup(
    name="PACKAGENAME",
    version=__version__,
    description="DESCRIPTION",
    url="URL",
    author="YOUR NAME",
    author_email="youremail@domain.com",
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
    install_requires=["numpy>=1.11"],
    entry_points={"console_scripts": ["CLINAME=PACKAGENAME._cli:main"]},
    zip_safe=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
