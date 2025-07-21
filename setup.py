from setuptools import setup, find_packages

setup(
    name='rul_timewarping',
    version='0.1',
    packages=find_packages(),  # finds rul_timewarping automatically
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'mpmath',
        'pytest',
        'pytest-cov',
        'black',
    ],
    python_requires='>=3.7',
)