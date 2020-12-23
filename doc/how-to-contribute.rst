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

To test the docs locally requires Python >= 3.5:
      - ``$ pip install sphinx sphinx_rtd_theme``
      - ``$ python setup.py build_doc``

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


Using Docker for development
----------------------------

Docker containers are useful for developing, testing and debugging PyTango.  See the
folder ``.devcontainer`` in the root of the source repo.  It includes instructions for
building the Docker images and using them for development.

For direct usage, rather than PyTango developement, a Docker image with PyTango already
installed is available:  https://hub.docker.com/r/tangocs/tango-pytango.

Releasing a new version
-----------------------

From time to time a new version should be released.  Anyone who wishes to see some
features of the development branch released is free to make a new release.  The basic
steps required are as follows:

Pick a version number
  * Semantic version numbering is used:  <major>.<minor>.<patch>
  * The major and minor version fields (9.3) track the TANGO C++ core version.
  * Small changes are done as patch releases.  For these the version
    number should correspond the current development number since each
    release process finishes with a version bump.
  * Patch release example:
      - ``9.3.3.devN`` or ``9.3.3aN`` or ``9.3.3bN`` (current development branch)
      - changes to ``9.3.3`` (the actual release)
      - changes to ``9.3.4.dev0`` (bump the patch version at the end of the release process)

Create an issue in Github
  * This is to inform the community that a release is planned.
  * Use a checklist similar to the one below:

    | Task list:
    | - [ ] Read steps in the how-to-contribute docs for making a release
    | - [ ] Pull request to update changelog and bump version
    | - [ ] Merge PR (this is the last PR for the release)
    | - [ ] Merge develop into stable
    | - [ ] Make sure Travis and Appveyor are OK on stable branch
    | - [ ] Make sure the documentation is updated for stable (readthedocs)
    | - [ ] Create an annotated tag from stable branch
    | - [ ] Make sure the documentation is updated for release (readthedocs)
    | - [ ] Upload the new version to PyPI
    | - [ ] Bump the version with "-dev" in the develop branch
    | - [ ] Create and fill in the release description on GitHub
    | - [ ] Build conda packages
    | - [ ] Advertise the release on the mailing list
    | - [ ] Close this issue

  * A check list in this form on github can be ticked off as the work progresses.

Make a branch from ``develop`` to prepare the release
  * Example branch name: ``prepare-v9.3.3``.
  * Edit the changelog (in ``docs/revision.rst``).  Include *all* pull requests
    since the version was bumped after the previous release.
  * Bump the versions (``tango/release.py`` and ``appveyor.yml``).
    E.g. ``version_info = (9, 3, 3)``, and ``version: 9.3.3.{build}``
  * Create a pull request to get these changes reviewed before proceeding.

Merge ``stable`` into ``develop``
  * Wait until the preparation branch pull request has been merged.
  * Merge ``stable`` into the latest ``develop``.  It is recommended to do a
    fast-forward merge in order to avoid a confusing merge commit. This can be
    done by simply pushing ``develop`` to ``stable`` using this command:

      ``$ git push origin develop:stable``

    This way the release tag corresponds to the actual release commit both on the
    ``stable`` and ``develop`` branches.
  * In general, the ``stable`` branch should point to the latest release.

Make sure Travis and AppVeyor are OK on ``stable`` branch
  * On Travis, all tests, on all versions of Python must be passing.
    If not, bad luck - you'll have to fix it first, and go back a few steps...
  * On AppVeyor, all builds, on all versions of Python must be passing.
    If not, bad luck - you'll have to fix it first, and go back a few steps...

Make sure the documentation is updated
  * Log in to https://readthedocs.org.
  * Get account permissions for https://readthedocs.org/projects/pytango from another
    contributor, if necessary.
  * Readthedocs *should* automatically build the docs for each:
      - push to develop (latest docs)
      - push to stable (stable docs)
      - new tags (e.g v9.3.3)
  * *But*, the webhooks are somehow broken, so it probably won't work automatically!
      - Trigger the builds manually here:  https://readthedocs.org/projects/pytango/builds/
      - Set the new version to "active" here:
        https://readthedocs.org/dashboard/pytango/versions/

Create an annotated tag for the release
  * Note: Github's release page makes lightweight tags which we don't want
  * Create tag:
      - ``$ git checkout stable``
      - ``$ git pull``
      - ``$ git tag -a -m "tag v9.3.3" v9.3.3``
      - ``$ git push -v origin refs/tags/v9.3.3``

Upload the new version to PyPI
  * Log in to https://pypi.org.
  * Get account permissions for PyTango from another contributor, if necessary.
  * If necessary, pip install twine: https://pypi.org/project/twine/)
  * Build release from the tagged commit:
      - ``$ git clean -xfd  # Warning - remove all non-versioned files and directories``
      - ``$ git fetch``
      - ``$ git checkout v9.3.3``
      - ``$ python setup.py sdist``
  * Optional:  Upload to https://test.pypi.org, and make sure all is well:
      - ``$ twine upload -r testpypi dist/pytango-9.3.3.tar.gz``
  * Optional:  Test installation (in a virtualenv):
      - ``$ pip install -i https://test.pypi.org/simple/ pytango``
  * Upload the source tarball to the real PyPI:
      - ``$ twine upload dist/pytango-9.3.3.tar.gz``
  * Run build for the tag on AppVeyor, download artifacts, and upload wheels:
      - ``$ twine upload dist/pytango-9.3.3-cp27-cp27m-win32.whl``
      - ``$ twine upload dist/pytango-9.3.3-cp27-cp27m-win_amd64.whl``
      - ``$ twine upload dist/pytango-9.3.3-cp36-cp36m-win32.whl``
      - ``$ twine upload dist/pytango-9.3.3-cp36-cp36m-win_amd64.whl``
      - ``$ twine upload dist/pytango-9.3.3-cp37-cp37m-win32.whl``
      - ``$ twine upload dist/pytango-9.3.3-cp37-cp37m-win_amd64.whl``

Bump the version with "-dev" in the develop branch
  * Make branch like ``bump-dev-version`` from head of ``develop``.
  * Change all references to next version and next + 1.  E.g. if releasing
    v9.3.3, then update v9.3.4 to v9.3.5 and v9.3.3 to v9.3.4.
  * This includes files like ``README.rst``, ``doc/howto.rst``, ``doc/start.rst``, ``doc/how-to-contribute.rst``.
  * In ``tango/release.py``, change ``version_info``, e.g. from ``(9, 3, 3)`` to
    ``(9, 3, 4, 'dev', 0)``.
  * In ``appveyor.yml``, change ``version``, e.g. from ``9.3.3.{build}`` to
    ``9.3.4.dev0.{build}``.
  * Create PR, merge to ``develop``.

Create and fill in the release description on GitHub
  * Go to the Tags page: https://github.com/tango-controls/pytango/tags
  * Find the tag created above and click "Create release".
  * Content must be the same as the details in the changelog.  List all the
    pull requests since the previous version.

Build conda packages
  * This is tricky, so ask a contributor from the ESRF to do it.

Advertise the release on the mailing list
  * Post on the Python development list.
  * Example of a previous post:  http://www.tango-controls.org/community/forum/c/development/python/pytango-921-release/

Close off release issue
  * All the items on the check list should be ticked off by now.
  * Close the issue.
