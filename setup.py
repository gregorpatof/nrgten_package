from setuptools import Extension
from setuptools.command.install import install
import setuptools
import subprocess
from distutils.command.install import install as _install
import platform
import os

class CustomInstall(install):
    def run(self):
        # command = "gcc -shared -o vcon_lib_python.so -fPIC Vcontacts-v1.2_for_python.c"
        command = "git clone https://github.com/gregorpatof/vcontacts"
        process = subprocess.Popen(command, shell=True, cwd="nrgens")
        process.wait()

        install.run(self)

module_ext = Extension('nrgens.vcontacts',
                   sources = ['nrgens/vcontacts/Vcontacts-v1.2_for_python.c'],
                   include_dirs = ['nrgens/vcontacts'],
                       )
                   # extra_compile_args=['-fPIC'])

class install(_install):
    def run(self):
        op_system = platform.system()
        args = []
        args2 = None
        if op_system == "Windows":
            args = ['cl.exe', '/LD', 'nrgens\\vcontacts\\Vcontacts-v1.2_for_python.c']
            args2 = ['cp', 'Vcontacts-v1.2_for_python.dll', 'nrgens\\vcontacts\\vconlib.dll']
        elif op_system == "Darwin" or op_system == "Linux":
            args = ['gcc',  '-shared', '-o', 'nrgens/vcontacts/vconlib.so',
                    '-fPIC', 'nrgens/vcontacts/Vcontacts-v1.2_for_python.c']
        else:
            raise ValueError("Unrecognized operating system: {0}".format(op_system))
        subprocess.call(args)
        if args2 is not None:
            subprocess.call(args2)
        _install.run(self)

setuptools.setup(
    name="nrgens", # Replace with your own username
    version="0.1",
    author="Gregor Patof",
    description="Najmanovich Research Group Elastic NetworkS",
    packages=setuptools.find_packages(),
    package_data={'nrgens': ['config/*.atomtypes', 'config/*.masses', 'vcontacts/*.c', 'vcontacts/*.h']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # cmdclass={'install': install},
    # ext_modules=[module_ext],
    python_requires='>=3.6',
)