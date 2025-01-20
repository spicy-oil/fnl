from setuptools import setup, find_packages

setup(
    name='fts-nn-ll',
    version='0.1.0',
    description='A neural network package for spectral line detection in high-resolution FT atomic emission spectroscopy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Milan Ding',
    author_email='milan.ding15@imperial.ac.uk',
    url='https://github.com/spicy-oil/fts-nn-ll',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Versions used for the package development
        'numpy >= 1.26.4, < 2',
        'scipy >= 1.13.0, < 2',
        'matplotlib >= 3.8.4, < 4',
        'pandas >= 2.2.1, < 3', # for saving the linelist to Excel
        'openpyxl >= 3.1.2, < 4', # for saving the linelist to Excel
        'torch >= 2.4.0, < 3',
        'scikit-learn >= 1.4.2, < 2',
        'tqdm >= 4.66.2, < 5'
    ],
    extras_require={
        # torch was compiled with CUDA 12.4 for the package development
    },
    python_requires='>= 3.10',  # 3.10.12 for the package development
    classifiers=[
        'Programming Language :: Python :: 3.10'
    ]
)