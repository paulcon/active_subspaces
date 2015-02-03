# setup.py
# 
# Created:  Trent L., Feb 2014
# Modified:         

""" active_subspaces setup script
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import sys, os

# ----------------------------------------------------------------------
#   Main - Run Setup
# ----------------------------------------------------------------------

def main():
    
    the_package = 'active_subspaces'
    version = '0.0.0'
    date = 'February 02, 2013'
    
    if len(sys.argv) >= 2:
        command = sys.argv[1]
    else:
        command = ''
    
    if command == 'uninstall':
        uninstall(the_package,version,date)
    else:
        install(the_package,version,date)
 
 
# ----------------------------------------------------------------------
#   Install Pacakge
# ----------------------------------------------------------------------

def install(the_package,version,date):
    
    # imports
    try:
        from setuptools import setup, find_packages
    except ImportError:
        print 'setuptools required to install'
        sys.exit(1)
    
    # list all sub packages
    exclude = [ d+'.*' for d in os.listdir('.') if d not in [the_package] ]
    packages = find_packages( exclude = exclude )
    
    # run the setup!!!
    setup(
        name = the_package,
        version = version, 
        description = 'Utilities for discovering and exploiting active subspaces',
        author = 'Paul Constantine',
        author_email = '',
        maintainer = 'The Developers',
        url = '',
        packages = packages,
        license = '',
        platforms = ['Win, Linux, Unix, Mac OS-X'],
        zip_safe  = False,
        long_description = read('README')
    )  
    
    return


# ----------------------------------------------------------------------
#   Un-Install Package
# ----------------------------------------------------------------------

def uninstall(the_package,version,date):
    """ emulates command "pip uninstall"
        just for syntactic sugar at the command line
    """
    
    import sys, shutil
    
    # clean up local egg-info
    try:
        shutil.rmtree(the_package + '.egg-info')
    except:
        pass        
        
    # import pip
    try:
        import pip
    except ImportError:
        print 'pip is required to uninstall this package'
        sys.exit(1)
    
    # setup up uninstall arguments
    args = sys.argv
    del args[0:1+1]
    args = ['uninstall', the_package] + args
    
    # uninstall
    try:
        pip.main(args)
    except:
        pass
    
    return
    
    
# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------

def read(path):
    """Build a file path from *paths and return the contents."""
    with open(path, 'r') as f:
        return f.read()
    
# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()