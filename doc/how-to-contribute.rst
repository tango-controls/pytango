.. currentmodule:: tango

.. highlight:: python
   :linenothreshold: 3

.. _how-to-contribute:

=================
How to Contribute
=================

Everyone is welcome to contribute to PyTango project.
If you don't feel comfortable with writing core PyTango we are looking for contributors to documentation or/and tests.

Workflow
--------

A normal Git workflow is used. You can find how to automate your git branching workflow example_.
Good practices:

* There is no special policy regarding commit messages. They should be short (50 chars or less) and contain summary of all changes,
* A CONTRIBUTING file is required,
* Pull requests should be ALWAYS made to develop branch, not to a master branch.

reStructuredText and Sphinx
---------------------------

Documentation is written in reStructuredText_ and built with Sphinx_ - it's easy to contribute.
It also uses autodoc_ importing docstrings from tango package.
Theme is not important, a theme prepared for Tango Community can be also used.

Source code standard
--------------------

All code should be PEP8_ compatible. We have set up checking code quality with
Codacy_ which uses PyLint_ under the hood. You can see how well your code is
rated on your PR's page.

.. note:: The accepted policy is that your code **cannot** introduce more
          issues than it solves!

You can also use other tools for checking PEP8_ compliance for your
personal use. One good example of such a tool is Flake8_ which combines PEP8_
and PyFlakes_. There are plugins_ for various IDEs so that you can use your
favourite tool easily.



.. _example: http://jeffkreeftmeijer.com/2010/why-arent-you-using-git-flow
.. _autodoc: https://pypi.python.org/pypi/autodoc
.. _PEP8: https://www.python.org/dev/peps/pep-0008
.. _Flake8: https://gitlab.com/pycqa/flake8
.. _PyFlakes: https://github.com/PyCQA/pyflakes
.. _plugins: https://gitlab.com/pycqa/flake8/issues/286
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://www.sphinx-doc.org/en/stable
.. _PyLint: https://www.pylint.org
.. _Codacy: https://www.codacy.com/app/tango-controls/pytango/dashboard
