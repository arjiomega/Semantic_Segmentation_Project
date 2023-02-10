from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent

with open(Path(BASE_DIR,"requirements.txt"),"r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages = [
    "pytest == 7.2.1",
]

setup(
    name="Semantic Segmentation Project",
    version=0.1,
    description=" ",
    author="Richard Joseph Omega",
    author_email="richardjoseph.omega@gmail.com",
    url="https://github.com/arjiomega/Semantic_Segmentation_Project",
    python_requires=">=3.9",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extra_require = {
        "dev": test_packages,
        "test": test_packages,
    },
)