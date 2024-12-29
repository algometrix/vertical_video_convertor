"""Setup file for the face_analysis package."""

from setuptools import find_packages, setup

setup(
    name="face_analysis",
    version="0.1.0",
    description="Video conversion to vertical format using face detection",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "insightface",
        "opencv-python",
        "opencv-python-headless",
        "moviepy",
        "numpy",
    ],
) 