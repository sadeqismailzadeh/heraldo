from setuptools import setup, find_packages

setup(
    name="heraldo",
    version="0.1",
    packages=[p for p in find_packages() if p.startswith("heraldo")],
    install_requires=[
        "strawberryfields",
        ]
)