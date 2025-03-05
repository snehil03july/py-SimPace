from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='py-simpace',
    version='0.1.1',
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
    long_description=long_description,
    long_description_content_type="text/markdown",  # Ensure markdown format is used
    license='MIT',
    url='https://github.com/snehil03july/py-simpace',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
