from setuptools import setup, find_packages

setup(
    name="quantiq",
    version="0.1.1",
    description="Confidence-aware AI toolkit with uncertainty estimation for Transformers and Deep Ensembles",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Aryan Patil",
    author_email="aryanator01@gmail.com",
    url="https://github.com/aryanator/QuantIQ",
    project_urls={
        "Documentation": "https://github.com/aryanator/QuantIQ",
        "Source": "https://github.com/aryanator/QuantIQ",
        "Bug Tracker": "https://github.com/aryanator/QuantIQ/issues",
    },
    packages=find_packages(exclude=["tests", "__pycache__"]),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.30.0",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)