from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='bgl_prediction',
    version='0.1',
    description='Different approaches for blood glucose level prediction',
    author='Iara Ravagni',
    author_email='iara.ravagni@duke.edu',
    packages=find_packages(),  # Automatically find all the packages in your project
    install_requires=required,
    entry_points={  # Define command line entry points if necessary
        'console_scripts': [
            'make_dataset = scripts.make_dataset:main',  # Custom script to load data
            # 'build_features = scripts.build_features:main',  # Custom script to build features
            # 'train_model = scripts.train_model:main',  # Custom script to train model
        ],
    },
)