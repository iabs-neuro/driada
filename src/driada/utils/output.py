"""
Utilities for capturing and displaying stdout output.

This module provides tools for temporarily capturing stdout output during
code execution, useful for testing, logging, or capturing output from
third-party libraries.
"""

from io import StringIO
import sys
from typing import List, Any


class Capturing(list):
    """
    Context manager that captures stdout output into a list.
    
    Temporarily redirects sys.stdout to capture all print statements and
    stdout writes during the context. The captured output is stored as a
    list of lines (strings without newlines).
    
    Attributes
    ----------
    Inherits from list, so captured lines are accessible as list elements.
    
    Returns
    -------
    self : Capturing
        The Capturing instance itself, which is a list of captured lines.
        
    Warnings
    --------
    This class is NOT thread-safe. Using multiple Capturing contexts in
    different threads simultaneously will cause interference. 
    
    Nested usage may have unexpected behavior - inner contexts will restore
    stdout to the outer context's StringIO, not the original stdout.
    
    Examples
    --------
    >>> with Capturing() as output:
    ...     print("Hello")
    ...     print("World")
    >>> output
    ['Hello', 'World']
    
    Using with functions that print::
    
        with Capturing() as log:
            some_verbose_function()
        show_output(log)  # Display captured output
    
    Notes
    -----
    Memory usage scales with output size. For very large outputs, consider
    alternative approaches like writing to temporary files.    """
    def __enter__(self):
        """Enter the context manager and start capturing stdout.
        
        Returns
        -------
        Capturing
            Returns self to allow access to the captured output.        """
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        """Exit the context manager and restore stdout.
        
        Parameters
        ----------
        *args
            Exception information (type, value, traceback) if any.
            These are ignored, allowing exceptions to propagate.        """
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def show_output(output: List[str]) -> None:
    """
    Display captured output from Capturing context.
    
    Prints each line from the captured output to stdout. If the output
    is empty, prints a message indicating the log is empty.
    
    Parameters
    ----------
    output : list of str
        List of captured output lines, typically from Capturing context.
        Each element should be a string representing one line of output.
        
    Returns
    -------
    None
        
    Raises
    ------
    TypeError
        If output is None or not a list-like object.
    AttributeError
        If output doesn't support len() or iteration.
        
    Examples
    --------
    >>> with Capturing() as log:
    ...     print("Test output")
    >>> show_output(log)
    Test output
    
    >>> show_output([])
    log is empty
    
    >>> show_output(["Line 1", "Line 2"])
    Line 1
    Line 2
    
    Notes
    -----
    This function is designed to work with the output from Capturing
    context manager but can display any list of strings.    """
    if len(output) == 0:
        print("log is empty")
    else:
        for s in output:
            print(s)
