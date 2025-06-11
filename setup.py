from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py-simpace",
    version="2.0.0",
    author="Snehil Kumar",
    author_email="sk895@exeter.ac.uk",
    description="Realistic MRI motion artifact simulation toolkit for structural MRI and fMRI, with deep learning integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/snehil03july/py-SimPace",
    project_urls={
        "Bug Tracker": "https://github.com/snehil03july/py-SimPace/issues",
        "Documentation": "https://github.com/snehil03july/py-SimPace#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.5",
        "nibabel>=3.2",
        "torch>=1.10",
        "tqdm>=4.62",
        "scikit-image>=0.19",
    ],
    extras_require={
        "dev": ["pytest>=6.2", "torchio>=0.18"]
    },
    include_package_data=True,
    zip_safe=False,
)
