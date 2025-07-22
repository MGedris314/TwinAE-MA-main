from setuptools import setup

setup(
    name='TwinAE-MA',
    version='0.0.1',
    py_modules=['AutoEncoders'],  # Only include AutoEncoders.py
    description='Includes all classes and functions to build Twin Autoencoders for Manifold Alignment',
    author='Adam G. Rustad',
    url='https://github.com/JakeSRhodesLab/TwinAE-MA',  
    include_package_data=True,
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "scikit-learn",
        "scipy>=1.11.0"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
)
