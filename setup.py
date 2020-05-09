import os
from setuptools import setup, find_packages
from os.path import join as pjoin
PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = os.path.join('ssnmf', 'version.py')
with open(ver_file) as f:
    exec(f.read())

# Dependencies
with open('requirements.txt') as f:
    requirements = f.readlines()

REQUIRES = [t.strip() for t in requirements]

opts = dict(name='ssnmf',
            maintainer='Valentina Staneva',
            maintainer_email='vms16@uw.edu',
            description='Smooth and Sparse NMF',
            long_description='Palm implementation of sparse NMF with Tichonov regularization of the time difference of the coefficients',
            url='https://github.com/valentina-s/ss-nmf',
            download_url='DOWNLOAD_URL',
            license='MIT License',
            classifiers='CLASSIFIERS',
            author='AUTHOR',
            author_email='AUTHOR_EMAIL',
            platforms='PLATFORMS',
            version='VERSION',
            packages=find_packages(),
            package_data={'ssnmf':[pjoin('data','*')]},
            install_requires=REQUIRES,
            requires=REQUIRES)


if __name__ == '__main__':
    setup(**opts)
