from os import path
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = path.join('tsnmf', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# Dependencies
with open('requirements.txt') as f:
    requirements = f.readlines()

# Readme for description
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

REQUIRES = [t.strip() for t in requirements]

opts = dict(name='time-series-nmf',
            maintainer='Valentina Staneva',
            maintainer_email='vms16@uw.edu',
            description='Time Series NMF',
            long_description=long_description,
            long_description_content_type='text/markdown',
            url='https://github.com/valentina-s/time-series-nmf',
            license='MIT License',
            author='Valentina Staneva',
            author_email='vms16@uw.edu',
            version='0.1.0.dev0',
            packages=find_packages(),
            package_data={'tsnmf':[path.join('data','*')]},
            install_requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)
