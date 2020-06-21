with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metricks", # Replace with your own username
    version="0.0.1",
    author="Robert Turnbull",
    author_email="rob@robturnbull.com",
    description="Metrics and callbacks for Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rbturnbull/metricks",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
