from setuptools import setup, find_packages

setup(
    name="helchriss",
    version="1.0",
    author="Yiqi Sun (Zongjing Li)",
    author_email="ysun697@gatech.edu",
    description="helchriss, the personal package for machine learning and more",

    # project main page

    # the package that are prerequisites
    packages=find_packages(),
    include_package_data = True,
    package_data={
        'helchriss': ['helchriss/*.grammar']
        },
)

"""
'':['moic',
        'moic/mklearn',
        'moic/learn/nn'],
        'moic': ['mklearn'],
        'bandwidth_reporter':['moic','moic/mklearn','moic/learn/nn']
               
"""