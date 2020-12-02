
import setuptools

setuptools.setup(
    name="nrgten", # Replace with your own username
    version="0.1",
    author="Gregor Patof",
    description="Najmanovich Research Group Toolkit for Elastic Networks",
    packages=setuptools.find_packages('src'),
    package_dir={'':'src'},
    package_data={'nrgten': ['config/*.atomtypes', 'config/*.masses', 'vcontacts/*.c', 'vcontacts/*.h']},
    test_suite='tests',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)