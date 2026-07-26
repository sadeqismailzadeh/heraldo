from setuptools import setup, find_packages

setup(
    name="quantum_agent",
    version="0.1",
    packages=[p for p in find_packages() if p.startswith("quantum_agent")],
    install_requires=[
        "strawberryfields",
        ]
)