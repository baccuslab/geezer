from setuptools import setup, find_packages

setup(
    name='geezer',
    version='1.0',
    author='Joshua Melander, Javier Weddington, Youssef Faragalla',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'geezer=geezer.gui.main:main'
        ]
     },
    include_package_data=True,
    zip_safe=False,
)
