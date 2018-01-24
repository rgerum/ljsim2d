try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='ljsim2D',
      version='1.0',
      description='Simulation a 2D system with Lennard Jones interactions',
      license='GPLv3',
      author='Richard Gerum',
      packages=['ljsim2D'],
      install_requires=["numpy", "tqdm", "matplotlib"]
      )
