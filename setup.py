from setuptools import setup, find_packages

setup(
    name="quantum_agent",
    version="0.1",
    package_dir={"": "quantum_agent"},
    packages=find_packages(where="quantum_agent"),
    install_requires=[
        "gymnasium",
        "stable-baselines3",
        "strawberryfields",
        "sb3-contrib"
    ]
)