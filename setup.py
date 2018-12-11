from setuptools import setup, find_packages
# import sys
# if sys.version_info >= (3,7):
#     sys.exit('Sorry, Python >= 3.7 is not supported')

setup(
    name='MOBO',
    python_requires='<3.7',
    version='0.0.1',
    description='Sample package for Python-Guide.org',
    # long_description=readme,
    author='Masashi Sode',
    author_email='masashi.sode@gmail.com',
    install_requires=['numpy',
                      'matplotlib', 'scipy',
                      'scikit-learn',
                      'pygmo', 'sphinx',
                      'sphinxcontrib-seqdiag',
                      'sphinx_rtd_theme',
                      'recommonmark'],
    url='https://github.com/MasashiSode',
    # license=license,
    packages=find_packages(exclude=['tests', 'docs']),
)
