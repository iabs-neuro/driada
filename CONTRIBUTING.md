# Contributing to DRIADA

Thank you for your interest in contributing to DRIADA! We welcome contributions of all kinds - bug reports, feature requests, documentation improvements, and code contributions.

Please note that this project has a Code of Conduct. By participating, you agree to abide by its terms.

## Types of Contributions

We welcome many types of contributions:
- Bug reports and fixes
- New features or enhancements
- Documentation improvements
- Example notebooks and tutorials
- Test coverage improvements

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/driada.git
   cd driada
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=driada

# Run specific test directory
pytest tests/unit/experiment/
```

## Code Style

We use NumPy-style docstrings for documentation. Example:

```python
def example_function(param1: int, param2: str) -> bool:
    """Short description of the function.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> example_function(42, "test")
    True
    """
    pass
```

You can optionally format your code with Black and isort:

```bash
black src/driada tests
isort src/driada tests
```

## Building Documentation

To build and view documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build docs docs/_build/html
# Open docs/_build/html/index.html in your browser
```

## Submitting Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with a descriptive message:
   ```bash
   git commit -m "Add feature X"
   ```

3. Push to your fork and open a Pull Request on GitHub

### For Larger Contributions

If you're contributing significant new features or changes, we'd greatly appreciate if you could:
- Add tests for the new functionality (helps ensure everything works as expected)
- Run the test suite locally (`pytest`) to verify nothing breaks
- Update relevant documentation

This is optional but highly desirable, and helps maintain code quality and makes the review process smoother.

## Reporting Issues

Please check existing issues first to avoid duplicates.

When reporting bugs, please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Error messages (if applicable)

## Questions?

Feel free to open an issue or start a discussion if you have questions about the codebase or how to contribute.

## License

By contributing to DRIADA, you agree that your contributions will be licensed under the MIT License.
