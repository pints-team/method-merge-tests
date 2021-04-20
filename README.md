# method-merge-tests
Provides a house for test results for samplers when they are merged into PINTS. There is one Jupyter notebook per sampling method.

Note, functional tests examine the ongoing performance of methods. This repository instead houses the more extensive results of running samplers across multiple toy problems.

Any new method introduced into PINTS needs a Jupyter notebook with these tests as a prerequisite for it being merged into the master branch. Note, a number of existing methods were merged before this repository was created. For these methods, the tests were run post-merge.

Each Jupyter notebook should contain and graphically illustrate results across multiple numbers of iterations across a number of replicates.
