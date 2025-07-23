from setuptools import setup, find_packages

setup(
    name='rul_timewarping',
    version='0.2',
    packages=find_packages(),  # finds rul_timewarping automatically
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'mpmath',
        'pytest',
        'pytest-cov',
        'black',
        'lifelines',
    ],
    python_requires='>=3.7',
)