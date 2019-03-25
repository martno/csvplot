import setuptools

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'beautifulsoup4>=4.7.1',
    'Click>=7.0',
    'Flask>=1.0.2',
    'matplotlib>=3.0.3',
    'numpy>=1.16.2',
    'pandas>=0.24.2',
    'Pillow>=5.4.1',
    'seaborn>=0.9.0',
    'Werkzeug==0.14.1',  # https://stackoverflow.com/a/55297531
    'xlrd>=1.2.0',
]

setuptools.setup(
    name="csvplot",
    version="0.1.0",
    url="https://github.com/martno/csvplot/",

    author="Martin Nordstrom",
    author_email="martin.nordstrom87@gmail.com",

    description="Make statistical visualizations of CSV files",
    long_description=readme,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(include=['csvplot']),
    package_dir={'csvplot': 'csvplot'},
    package_data={
        'csvplot': [
            'css/*',
            'js/*',
            'static/*',
            'templates/*',
        ],
    },

    install_requires=requirements,
    license="MIT license",

    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts': [
            'csv-plot=csvplot.csv_plot:main',
        ],
    },
)
