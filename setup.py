import os.path
import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md")) as f:
    README = f.read()

setuptools.setup(
    name="nrgten",
    version="1.0.7",
    author="Olivier Mailhot",
    description="Najmanovich Research Group Toolkit for Elastic Networks",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/gregorpatof/nrgten_package",
    packages=setuptools.find_packages('src'),
    package_dir={'':'src'},
    package_data={'nrgten': ['config/*.atomtypes', 'config/*.masses', 'vcontacts/*.c', 'vcontacts/*.h']},
    test_suite='tests',
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)