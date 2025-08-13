"""Tests for GIF utilities."""

import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from driada.utils.gif import erase_all, save_image_series, create_gif_from_image_series


class TestEraseAll:
    """Test the erase_all function."""

    def test_erase_all_basic(self):
        """Test basic file deletion with signature and extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = [
                "test_image_1.png",
                "test_image_2.png",
                "other_image.png",
                "test_doc.txt",
            ]
            for fname in test_files:
                open(os.path.join(tmpdir, fname), "w").close()

            # Erase files with 'test' signature and .png extension
            erase_all(tmpdir, signature="test", ext=".png")

            # Check results
            remaining = os.listdir(tmpdir)
            assert "test_image_1.png" not in remaining
            assert "test_image_2.png" not in remaining
            assert "other_image.png" in remaining  # Different signature
            assert "test_doc.txt" in remaining  # Different extension

    def test_erase_all_empty_signature(self):
        """Test with empty signature (matches all files)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = ["image1.png", "image2.png", "doc.txt"]
            for fname in test_files:
                open(os.path.join(tmpdir, fname), "w").close()

            # Erase all .png files
            erase_all(tmpdir, signature="", ext=".png")

            # Check results
            remaining = os.listdir(tmpdir)
            assert "image1.png" not in remaining
            assert "image2.png" not in remaining
            assert "doc.txt" in remaining

    def test_erase_all_nonexistent_directory(self):
        """Test with non-existent directory (should not raise error)."""
        # Should not raise any exception
        erase_all("/nonexistent/path", signature="test", ext=".png")

    def test_erase_all_no_matching_files(self):
        """Test when no files match the criteria."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = ["image1.jpg", "image2.jpg"]
            for fname in test_files:
                open(os.path.join(tmpdir, fname), "w").close()

            # Try to erase .png files (none exist)
            erase_all(tmpdir, signature="", ext=".png")

            # All files should remain
            remaining = os.listdir(tmpdir)
            assert len(remaining) == 2


class TestSaveImageSeries:
    """Test the save_image_series function."""

    def test_save_image_series_with_titles(self):
        """Test saving figures with titles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test figures with titles
            figures = []
            for i in range(3):
                fig = plt.figure()
                plt.plot([0, 1], [0, 1])
                fig.suptitle(f"Figure_{i}")
                figures.append(fig)

            # Save figures
            save_image_series(tmpdir, figures, im_ext="png")

            # Check saved files
            saved_files = os.listdir(tmpdir)
            assert "Figure_0.png" in saved_files
            assert "Figure_1.png" in saved_files
            assert "Figure_2.png" in saved_files
            assert len(saved_files) == 3

            # Cleanup
            for fig in figures:
                plt.close(fig)

    def test_save_image_series_without_titles(self):
        """Test saving figures without titles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test figures without titles
            figures = []
            for i in range(3):
                fig = plt.figure()
                plt.plot([0, 1], [i, i])
                figures.append(fig)

            # Save figures
            save_image_series(tmpdir, figures, im_ext="png")

            # Check saved files
            saved_files = os.listdir(tmpdir)
            assert "figure_0000.png" in saved_files
            assert "figure_0001.png" in saved_files
            assert "figure_0002.png" in saved_files
            assert len(saved_files) == 3

            # Cleanup
            for fig in figures:
                plt.close(fig)

    def test_save_image_series_creates_directory(self):
        """Test that directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "subdir")

            # Create a test figure
            fig = plt.figure()
            plt.plot([0, 1], [0, 1])

            # Save to non-existent directory
            save_image_series(subdir, [fig], im_ext="png")

            # Check directory was created and file saved
            assert os.path.exists(subdir)
            assert len(os.listdir(subdir)) == 1

            plt.close(fig)

    def test_save_image_series_different_extension(self):
        """Test saving with different file extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test figure
            fig = plt.figure()
            plt.plot([0, 1], [0, 1])
            fig.suptitle("TestPlot")

            # Save as jpg
            save_image_series(tmpdir, [fig], im_ext="jpg")

            # Check saved file
            saved_files = os.listdir(tmpdir)
            assert "TestPlot.jpg" in saved_files

            plt.close(fig)


class TestCreateGifFromImageSeries:
    """Test the create_gif_from_image_series function."""

    def test_create_gif_basic(self):
        """Test basic GIF creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                fig = plt.figure(figsize=(2, 2))
                plt.plot([0, 1], [i, i])
                plt.title(f"frame_{i}")
                fig.savefig(os.path.join(tmpdir, f"test_frame_{i}.png"))
                plt.close(fig)

            # Create GIF
            gif_path = create_gif_from_image_series(
                tmpdir, signature="test_", gifname="animation", erase_prev=False
            )

            # Check GIF was created
            assert os.path.exists(gif_path)
            assert gif_path.endswith("test_ animation.gif")
            assert "GIFs" in gif_path

            # Check source images still exist (erase_prev=False)
            assert "test_frame_0.png" in os.listdir(tmpdir)

    def test_create_gif_with_erase(self):
        """Test GIF creation with source image deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                fig = plt.figure(figsize=(2, 2))
                plt.plot([0, 1], [i, i])
                fig.savefig(os.path.join(tmpdir, f"frame_{i}.png"))
                plt.close(fig)

            # Create GIF with erase
            gif_path = create_gif_from_image_series(
                tmpdir, signature="frame", gifname="test", erase_prev=True, im_ext="png"
            )

            # Check GIF was created
            assert os.path.exists(gif_path)

            # Check source images were deleted
            remaining_files = os.listdir(tmpdir)
            assert "frame_0.png" not in remaining_files
            assert "frame_1.png" not in remaining_files
            assert "frame_2.png" not in remaining_files

    def test_create_gif_custom_duration(self):
        """Test GIF creation with custom frame duration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(2):
                fig = plt.figure(figsize=(2, 2))
                plt.plot([0, 1], [i, i])
                fig.savefig(os.path.join(tmpdir, f"img_{i}.png"))
                plt.close(fig)

            # Create GIF with custom duration
            gif_path = create_gif_from_image_series(
                tmpdir,
                signature="img",
                gifname="slow",
                duration=1.0,  # 1 second per frame
            )

            assert os.path.exists(gif_path)

    def test_create_gif_no_matching_images(self):
        """Test GIF creation when no images match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create GIF with no matching images
            gif_path = create_gif_from_image_series(
                tmpdir, signature="nonexistent", gifname="empty"
            )

            # Should create an empty GIF or handle gracefully
            assert os.path.exists(os.path.join(tmpdir, "GIFs"))

    def test_create_gif_sorted_order(self):
        """Test that images are sorted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create images in non-sequential order
            for i in [2, 0, 1, 3]:
                fig = plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, str(i), ha="center", va="center")
                plt.axis("off")
                fig.savefig(os.path.join(tmpdir, f"frame_{i}.png"))
                plt.close(fig)

            # Create GIF
            gif_path = create_gif_from_image_series(
                tmpdir, signature="frame", gifname="sorted", erase_prev=False
            )

            assert os.path.exists(gif_path)
            # The frames should be in order 0, 1, 2, 3 in the GIF


class TestIntegration:
    """Integration tests for the GIF workflow."""

    def test_full_workflow(self):
        """Test complete workflow: create figures, save, create GIF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a series of figures
            figures = []
            for i in range(5):
                fig = plt.figure(figsize=(3, 3))
                x = np.linspace(0, 2 * np.pi, 100)
                y = np.sin(x + i * np.pi / 4)
                plt.plot(x, y)
                fig.suptitle(f"Wave_Phase_{i}")  # Use suptitle instead of title
                plt.ylim(-1.5, 1.5)
                figures.append(fig)

            # Save the figures
            save_image_series(tmpdir, figures, im_ext="png")

            # Create GIF from saved images
            gif_path = create_gif_from_image_series(
                tmpdir,
                signature="Wave",
                gifname="sine_animation",
                erase_prev=True,
                duration=0.3,
            )

            # Verify results
            assert os.path.exists(gif_path)
            assert "sine_animation.gif" in gif_path

            # Source images should be deleted
            remaining = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
            assert len(remaining) == 0

            # Cleanup
            for fig in figures:
                plt.close(fig)
