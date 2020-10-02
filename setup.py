from distutils.core import setup

DESCRIPTION = "Branching Gaussian process."
LONG_DESCRIPTION = DESCRIPTION
NAME = "BranchedGP"
AUTHOR = "Alexis Boukouvalas"
AUTHOR_EMAIL = "alexis.boukouvalas@gmail.com"
MAINTAINER = "Alexis Boukouvalas"
MAINTAINER_EMAIL = "alexis.boukouvalas@gmail.com"
DOWNLOAD_URL = 'https://github.com/ManchesterBioinference/BranchedGP'
LICENSE = 'MIT'

VERSION = '2.0'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=DOWNLOAD_URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=['BranchedGP'],
    package_data={},
    install_requires=[
        "tensorflow>=2,<3",
        "gpflow>=2,<3",
        "matplotlib",
    ],
    python_requires=">=3.5",
)
