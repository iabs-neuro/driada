Contributing to DRIADA
======================

Thank you for your interest in contributing to DRIADA! We welcome contributions of all kinds - bug reports, feature requests, documentation improvements, and code contributions.

The full contribution guidelines are available in the repository:

.. include:: ../CONTRIBUTING.md
   :parser: myst_parser.sphinx_

Quick Links
-----------

- `Report a bug <https://github.com/iabs-neuro/driada/issues/new>`_
- `Request a feature <https://github.com/iabs-neuro/driada/issues/new>`_
- `View open issues <https://github.com/iabs-neuro/driada/issues>`_
- `Pull requests <https://github.com/iabs-neuro/driada/pulls>`_

Development Setup
-----------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/YOUR-USERNAME/driada.git
   cd driada
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=driada

   # Run specific test file
   pytest tests/test_experiment.py

Code Style
----------

DRIADA uses NumPy-style docstrings for documentation. You can optionally format your code with:

- **Black** for code formatting
- **isort** for import sorting

.. code-block:: bash

   black src/driada tests
   isort src/driada tests

Submitting Changes
------------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/your-feature-name``
3. Make your changes with descriptive commit messages
4. Push to your fork and open a Pull Request

For larger contributions, we'd greatly appreciate if you could:

- Add tests for new functionality (helps ensure everything works as expected)
- Run the test suite locally (``pytest``) to verify nothing breaks
- Update relevant documentation

This is optional but highly desirable and helps maintain code quality.

For complete details, see `CONTRIBUTING.md <https://github.com/iabs-neuro/driada/blob/main/CONTRIBUTING.md>`_.
