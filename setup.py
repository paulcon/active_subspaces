from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='active_subspaces',
      version='0.1.1',
      description='Tools to apply active subspaces to analyze their models and data.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
      ],
      keywords='math mathematics active subspaces',
      url='https://github.com/paulcon/active_subspaces',
      author='Paul Constantine',
      author_email='paul.constantine@mines.edu',
      license='MIT',
      packages=['active_subspaces', 'active_subspaces.qp_solvers', 'active_subspaces.utils'],
      install_requires=[
          'numpy',
          'scipy >= 0.12.0',
          'matplotlib'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
