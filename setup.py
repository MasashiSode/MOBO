from setuptools import setup, find_packages

setup(
    name='mobo',
    version='0.0.1',
    description='Multi-objective bayesian optimization package',
    # long_description=readme,
    author='Masashi Sode',
    author_email='masashi.sode@gmail.com',
    url='https://github.com/MasashiSode',
    # license=license,
    packages=find_packages(exclude=['tests', 'docs']),
)
