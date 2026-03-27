# Contributing to Active Learning for Emotion Classification

First off, thanks for taking the time to contribute!  By working together we can
build a better library for experimenting with active learning in emotion
classification.

The following is a set of guidelines for contributing to this project.  These are
mostly guidelines, not hard rules, and we try to be flexible.  If something
doesn’t make sense or seems inconsistent, please open an issue.

## Code of Conduct

This project and everyone participating in it is governed by the [Code of
Conduct](CODE_OF_CONDUCT.md).  By participating, you are expected to uphold
this code.  Please report unacceptable behaviour to the maintainers.

## How Can I Contribute?

### Reporting Bugs

If you find a bug in the code, please open an issue and provide a clear and
descriptive title.  Include as much relevant information as you can: your
operating system, Python version, steps to reproduce, expected behaviour and
actual behaviour.  If possible, include a minimal example or failing test.

### Suggesting Enhancements

We welcome suggestions for new features or improvements.  Before submitting a
proposal, please check the existing issues to see if someone has already
suggested it.  When opening a new enhancement request, provide a detailed
description of the problem you’re trying to solve and why it would be useful.

### Submitting Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you’ve added code that should be tested, add appropriate tests and
   ensure that all existing tests still pass.
3. Include thoughtful comments and docstrings explaining what your changes do.
4. Make sure your code follows a consistent style (PEP 8 for Python) and runs
   without errors on supported Python versions.
5. Write descriptive commit messages.  Follow the convention of summarising
   your change in the first line (50 characters or fewer) and providing more
   detail in the body if necessary.  For example:
   ```bash
   git commit -m "Add uncertainty sampling to active learning loop"
   ```
   Avoid vague messages like `Fix stuff`.
6. Submit a pull request against the `main` branch and describe your changes in
   the pull request description.  Link any related issues and mention
   maintainers for review.

### Branching Strategy

We follow a simple branching strategy: the `main` branch contains the stable
version of the code.  New work should be developed on feature branches that
branch off `main` and are merged back via pull requests.  When opening a pull
request, ensure that your branch is up to date with `main`.

## Style Guidelines

We aim for readability and maintainability.  Follow PEP 8 for Python code and
use tools such as `flake8` or `black` to check formatting.  Write docstrings
for all functions and classes.  Keep line lengths under 120 characters.

## Questions?

If you have questions about the project, please open an issue or start a
discussion.  We’re happy to help!
