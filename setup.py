from setuptools import setup, find_packages
import codecs
import os
 
classifiers = [
  'Development Status :: 1 - Planning',
  'Intended Audience :: Education',
  'Operating System :: Any',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='imgclassifier',
  version='0.0.1',
  description='A naive image classifier',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='https://github.com/mehedihasanbijoy/imgclassifier.git',  
  author='Joshua Lowe',
  author_email='mhb6434@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords=['image classifier', 'pytorch classifier', 'image recognition pytorch', 'image classification pytorch', 'classification', 'recognition', 'pytorch'], 
  packages=find_packages(),
  install_requires=['torch', 'torchvision'] 
)
