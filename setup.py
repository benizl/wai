from setuptools import setup
import subprocess, os

version = open('wai/version.txt').read().strip()

setup(
	name='wai',
	version=version,
	author='Ben Nizette',
	author_email='ben.nizette@liquidinstruments.com',
	packages=['wai'],
	package_dir={'wai': 'wai'},
	package_data={
		'wai' : ['version.txt']
	},
	license='MIT',
	long_description="Python scripting interface to the Liquid Instruments Moku:Lab",

	url="https://github.com/liquidinstruments/wai",
	download_url="https://github.com/liquidinstruments/wai/archive/%s.tar.gz" % version,

	keywords=['moku', 'liquid instruments', 'test', 'measurement', 'lab', 'equipment'],

	install_requires=[
		'numpy>=1.11.0',
        'scipy',
	],
)
