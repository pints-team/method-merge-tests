# Extensive tests for PINTS methods before merge
Provides a house for test results for samplers when they are merged into PINTS. There is one Jupyter notebook per sampling method.

Note, functional tests examine the ongoing performance of methods. This repository instead houses the more extensive results of running samplers across multiple toy problems.

Any new method introduced into PINTS needs a Jupyter notebook with these tests as a prerequisite for it being merged into the master branch. Note, a number of existing methods were merged before this repository was created. For these methods, the tests were run post-merge.

Each Jupyter notebook should contain and graphically illustrate results across multiple numbers of iterations across a number of replicates.


### How to use
Create a virtual environment with `python3 -m venv venv`, activate it with `source venv/bin/activate`, and install dependencies with `pip install -r requirements.txt`.
It also requires a local installation of [PINTS](https://github.com/pints-team/pints).
Since this repository is designated to test methods _before_ they are merged, these methods should be in a branch.
Therefore it is not included in the `requirements.txt` but one should checkout the corresponding branch when running the tests.

For now, the functions for these tests sit on [a branch in PINTS](https://github.com/pints-team/pints/tree/issue-1294-functional-testing-module) which needs to be checked out manually (and locally).
We don't have to do this when it's merged into master later.
