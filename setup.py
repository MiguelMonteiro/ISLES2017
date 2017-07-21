from setuptools import find_packages, setup

required_packages = ['cython', 'numpy', 'tensorflow', 'scipy']


setup(name='trainer',
      version='0.1',
      packages=['trainer'],
      install_requires=required_packages,
      include_package_data=True,
      description='ISLES 2017')
