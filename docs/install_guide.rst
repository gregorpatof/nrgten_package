Installation Guide
==================

macOS and Linux users
---------------------

To install the NRGTEN package, simply type::

	pip install nrgten

inside your terminal.

If you want to make sure the package works well, type::

	git clone https://github.com/gregorpatof/nrgten_examples

Followed by::

	cd nrgten_examples
	python simple_test.py

And you should see this appear in your terminal::

	NRGTEN is properly installed on your system!

Congratulations, you have now installed the NRGTEN package. Everything described
in this guide will work on your machine.

.. note::
	
	NRGTEN requires Python version 3.6 or greater to work. If your default Python installation is Python 2.x, you will need
	to use **pip3** instead of **pip** and **python3** instead of **python**.

Windows users
-------------

To install NRGTEN on Windows, some additional steps have to be taken. First, you
will need to install the `Build Tools for Visual Studio 2019 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_.

After the installation, search for an executable named **VS2015 x64 Native Tools
Command Prompt** and start it. Navigate to a desired folder and type::

	pip install nrgten

inside the terminal. Contrary to the macOS and Linux instructions, it is
necessary to clone the nrgten_examples repository::

	git clone https://github.com/gregorpatof/nrgten_examples

And then run the following example::

	cd nrgten_examples
	python simple_test.py

As above, you will see this message which confirms proper installation of NRGTEN::

	NRGTEN is properly installed on your system!

At this point, you can run NRGTEN from anywhere, including the standard **cmd**
executable. The **VS2015 x64 Native Tools Command Prompt** is only necessary
for the first run after installation (which compiles a C library).
