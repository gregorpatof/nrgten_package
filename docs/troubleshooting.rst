Troubleshooting
===============

Python version
--------------

NRGTEN requires Python 3.6 or higher. If you see an error message that looks like::

    Could not find a version that satisfies the requirement nrgten (from version: )
    No matching distribution found for nrgten

It probably means that you are using Python 2.x. You can try the installation command
using **pip3** instead of **pip** and **python3** instead of **python** to run the examples.
If you do not have these commands installed, you need to install Python 3 (https://www.python.org/downloads/).

If you see this message::

    There was a problem with the automatic compilation of Vcontacts. See the
    Troubleshooting section of the NRGTEN online documentation (nrgten.readthedocs.io)

It means that you are executing ENCoM for the first time and the pyvcon package (which
wraps around a legacy C executable that ENCoM needs) is unable to compile it, most
probably due to restrictive permissions on your system. Try rerunning with sudo,
installing locally instead of system-wide, or contact your sysadmin.
