"""Define cryoPPP dataset and EMPIAR dataset."""

import tarfile
from pathlib import Path
import wget


class CryoPPPData:
    """
    Class to handle CryoPPP dataset.

    Args:
        empiar_id (int): EMPIAR ID of the dataset.
        directory (str): Directory to store the dataset.

    Attributes:
        empiar_id (int): EMPIAR ID of the dataset.
        directory (Path): Directory to store the dataset.
        data_directory_url (str): Base URL for dataset.
        _data_tar_gz (Path): Path to downloaded tar.gz file.
    """

    def __init__(self, empiar_id: int, directory: str) -> None:
        self.empiar_id: int = empiar_id
        self.directory = Path(directory)
        self.data_directory_url = "https://calla.rnet.missouri.edu/cryoppp"
        self._data_tar_gz: Path = self.directory.joinpath(".cache", f"{self.empiar_id}.tar.gz")
        self.micrographs

    @property
    def micrographs(self) -> Path:
        """Property to access micrographs directory."""
        return self._get_directory("micrographs")

    @property
    def particles_stack(self) -> Path:
        """Property to access particles stack directory."""
        return self._get_directory("particles_stack")

    @property
    def ground_truth(self) -> Path:
        """Property to access ground truth directory."""
        return self._get_directory("ground_truth")

    def _get_directory(self, directory_name) -> Path:
        """Check if directory exists, if not, extract it."""
        directory: Path = self.directory.joinpath(str(self.empiar_id), directory_name)
        if not directory.is_dir():
            self.download_and_extract()
        return directory

    def _download(self) -> None:
        """Download EMPIAR dataset."""
        data_url: str = f"{self.data_directory_url}/{self.empiar_id}.tar.gz"
        self.directory.joinpath(".cache").mkdir(exist_ok=True)
        if self._data_tar_gz.exists():
            print(f"Using cache at {self._data_tar_gz}.")
        else:
            print(f"Downloading {self.empiar_id}.tar.gz from {self.data_directory_url}.")
            wget.download(data_url, out=str(self._data_tar_gz))

    def _extract(self) -> None:
        """Extract EMPIAR dataset."""
        if not self.directory.is_dir():
            self.directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(self._data_tar_gz, 'r:gz') as tar:
            print(f"Extracting {self.empiar_id}.tar.gz.")
            tar.extractall(path=self.directory.joinpath(str(self.empiar_id)))

    def download_and_extract(self) -> None:
        """Download and extract EMPIAR dataset."""
        self._download()
        self._extract()

    def __repr__(self) -> str:
        return f"CryoPPPData(empiar_id={self.empiar_id}, directory={self.directory})"