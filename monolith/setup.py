import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="monolith_lib",
    version="1.0.6",
    author="Du Mingzhe",
    author_email="dumingzhex@gmail.com",
    description="A code execution engine",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Elfsong/Monolith",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)