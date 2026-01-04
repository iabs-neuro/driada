"""Tests for output utilities."""

import sys
from driada.utils.output import Capturing, show_output


class TestCapturing:
    """Test the Capturing context manager."""

    def test_capture_single_print(self):
        """Test capturing a single print statement."""
        with Capturing() as output:
            print("Hello, World!")

        assert output == ["Hello, World!"]

    def test_capture_multiple_prints(self):
        """Test capturing multiple print statements."""
        with Capturing() as output:
            print("Line 1")
            print("Line 2")
            print("Line 3")

        assert output == ["Line 1", "Line 2", "Line 3"]

    def test_capture_empty(self):
        """Test capturing with no output."""
        with Capturing() as output:
            pass

        assert output == []

    def test_capture_with_newlines(self):
        """Test capturing output with embedded newlines."""
        with Capturing() as output:
            print("Line 1\nLine 2")

        assert output == ["Line 1", "Line 2"]

    def test_stdout_restored(self):
        """Test that stdout is properly restored after capture."""
        original_stdout = sys.stdout

        with Capturing() as output:
            print("Captured")

        assert sys.stdout is original_stdout

    def test_capture_exception_handling(self):
        """Test that stdout is restored even if exception occurs."""
        original_stdout = sys.stdout

        try:
            with Capturing() as output:
                print("Before exception")
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert sys.stdout is original_stdout
        assert output == ["Before exception"]

    def test_capture_extends_list(self):
        """Test that Capturing extends the list it inherits from."""
        capture = Capturing()
        capture.append("Initial")

        with capture:
            print("Captured")

        assert "Initial" in capture
        assert "Captured" in capture


class TestShowOutput:
    """Test the show_output function."""

    def test_show_empty_output(self, capsys):
        """Test showing empty output."""
        show_output([])
        captured = capsys.readouterr()
        assert captured.out == "log is empty\n"

    def test_show_single_line(self, capsys):
        """Test showing single line output."""
        show_output(["Single line"])
        captured = capsys.readouterr()
        assert captured.out == "Single line\n"

    def test_show_multiple_lines(self, capsys):
        """Test showing multiple lines output."""
        lines = ["Line 1", "Line 2", "Line 3"]
        show_output(lines)
        captured = capsys.readouterr()
        assert captured.out == "Line 1\nLine 2\nLine 3\n"

    def test_show_output_with_empty_strings(self, capsys):
        """Test showing output with empty strings."""
        lines = ["Line 1", "", "Line 3"]
        show_output(lines)
        captured = capsys.readouterr()
        assert captured.out == "Line 1\n\nLine 3\n"


class TestIntegration:
    """Test integration of Capturing and show_output."""

    def test_capture_and_show(self, capsys):
        """Test capturing output and then showing it."""
        with Capturing() as output:
            print("Captured line 1")
            print("Captured line 2")

        show_output(output)
        captured = capsys.readouterr()
        assert captured.out == "Captured line 1\nCaptured line 2\n"
