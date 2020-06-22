import os
from setuptools import setup, find_packages
from os.path import join as pjoin
PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = os.path.join('tsnmf', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# Dependencies
with open('requirements.txt') as f:
    requirements = f.readlines()

REQUIRES = [t.strip() for t in requirements]

opts = dict(name='tsnmf',
            maintainer='Valentina Staneva',
            maintainer_email='vms16@uw.edu',
            description='Time Series NMF',
            long_description='Palm implementation of sparse NMF with Tichonov regularization of the time difference of the coefficients',
            url='https://github.com/valentina-s/time-series-nmf',
            download_url='DOWNLOAD_URL',
            license='MIT License',
            classifiers='CLASSIFIERS',
            author='AUTHOR',
            author_email='AUTHOR_EMAIL',
            platforms='PLATFORMS',
            version='0.0.1.dev0',
            packages=find_packages(),
            package_data={'tsnmf':[pjoin('data','*')]},
            install_requires=REQUIRES,
            requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)
