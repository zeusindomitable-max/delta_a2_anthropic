from setuptools import setup, find_packages

setup(
    name="delta-a2-alignment-toolkit",
    version="1.0.0",
    author="Zeus Indomitable Max",
    author_email="zeusindomitablemax@protonmail.com",
    description="Δa₂ Alignment & Introspection Research Toolkit (AnthropicAI-Ready Edition)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zeusindomitable-max/delta-a2-alignment-toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "matplotlib>=3.8.0",
        "scipy>=1.11.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "delta-a2=src.cli.main:main",
        ],
    },
    keywords=[
        "alignment",
        "introspection",
        "anthropic",
        "LLM",
        "Δa2",
        "AI safety",
        "interpretability",
        "research toolkit",
    ],
)
