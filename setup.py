from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = f.read()

setup(
    name="nlplib", packages=find_packages(), install_requires=reqs.strip().split("\n")
)
