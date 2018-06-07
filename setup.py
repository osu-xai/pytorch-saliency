from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["torch==0.4.0"]


setup( name='pytorch-saliency',
       version='0.1',
       description='Pytorch plugin to generate saliencies',
       author='Magesh Kumar',
       author_email='m.magesh.66@gmail.com',
       include_package_data=False,
       package_data={},
       packages=find_packages(),
       install_requires=REQUIRED_PACKAGES
     )
