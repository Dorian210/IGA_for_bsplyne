from setuptools import setup, find_packages

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='IGA_for_bsplyne', 
    version='1.0.0', 
    url='https://github.com/Dorian210/IGA', 
    author='Dorian Bichet', 
    author_email='dbichet@insa-toulouse.fr', 
    description='A package for IGA elasticity.', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    packages=find_packages(), 
    install_requires=['numpy', 'scipy', 'tqdm', 'numba'], 
    classifiers=['Programming Language :: Python :: 3', 
                 'Operating System :: OS Independent', 
                 'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)'], 
    
)
