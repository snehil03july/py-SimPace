## setup.py

from setuptools import setup, find_packages

setup(
    name='py-simpace',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'nibabel',
        'pydicom',
        'matplotlib',
        'opencv-python'
    ],
    author='Snehil',
    description='A Python equivalent of SimPACE for simulating MRI motion artifacts.',
    license='MIT',
    url='https://github.com/snehil03july/py-simpace',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)