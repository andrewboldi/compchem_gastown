"""QMLForge Data Module - Dataset Management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
from ase import Atoms
from ase.db import connect

logger = logging.getLogger(__name__)


@dataclass
class QMDataPoint:
    """Single data point containing QM calculation results.

    Attributes:
        atoms: ASE Atoms object
        energy: Total energy in eV
        forces: Atomic forces (N_atoms, 3) in eV/Å
        stress: Stress tensor (3, 3) in eV/Å³ (optional)
        method: QM method used
        basis: Basis set used
        metadata: Additional metadata
    """

    atoms: Atoms
    energy: float
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    method: str = ""
    basis: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data."""
        if self.forces is not None:
            assert self.forces.shape == (len(self.atoms), 3), (
                f"Forces shape mismatch: {self.forces.shape} vs ({len(self.atoms)}, 3)"
            )


class QMLDataset:
    """Dataset manager for QMLForge.

    Stores and manages molecular datasets with QM data (energies, forces).
    Supports HDF5 and ASE database backends.

    Example:
        >>> dataset = QMLDataset("my_data.h5")
        >>> dataset.add(data_point)
        >>> for dp in dataset:
        ...     print(dp.energy)
    """

    def __init__(self, path: str | Path, backend: str = "hdf5", mode: str = "a"):
        """Initialize dataset.

        Args:
            path: Path to dataset file
            backend: Storage backend ("hdf5" or "ase")
            mode: File mode ("r", "w", "a")
        """
        self.path = Path(path)
        self.backend = backend
        self.mode = mode

        if backend == "hdf5":
            self._init_hdf5()
        elif backend == "ase":
            self._init_ase()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        logger.info(f"Initialized {backend} dataset: {path}")

    def _init_hdf5(self):
        """Initialize HDF5 backend."""
        self._file = h5py.File(self.path, self.mode)

        # Create groups if not exist
        if "structures" not in self._file:
            self._file.create_group("structures")
        if "energies" not in self._file:
            self._file.create_group("energies")
        if "forces" not in self._file:
            self._file.create_group("forces")
        if "metadata" not in self._file:
            self._file.create_group("metadata")

    def _init_ase(self):
        """Initialize ASE database backend."""
        self._db = connect(str(self.path))

    def add(self, data: QMDataPoint) -> int:
        """Add data point to dataset.

        Args:
            data: QMDataPoint to add

        Returns:
            Index of added data point
        """
        if self.backend == "hdf5":
            return self._add_hdf5(data)
        else:
            return self._add_ase(data)

    def _add_hdf5(self, data: QMDataPoint) -> int:
        """Add to HDF5."""
        idx = len(self._file["structures"])
        group = self._file["structures"].create_group(str(idx))

        # Store atomic numbers and positions
        group.create_dataset("numbers", data=data.atoms.numbers)
        group.create_dataset("positions", data=data.atoms.positions)

        # Store cell if periodic
        if data.atoms.pbc.any():
            group.create_dataset("cell", data=data.atoms.cell.array)
            group.create_dataset("pbc", data=data.atoms.pbc)

        # Store energy
        self._file["energies"].create_dataset(str(idx), data=data.energy)

        # Store forces if available
        if data.forces is not None:
            self._file["forces"].create_dataset(str(idx), data=data.forces)

        # Store metadata
        meta = {"method": data.method, "basis": data.basis, **data.metadata}
        self._file["metadata"].attrs[str(idx)] = json.dumps(meta)

        return idx

    def _add_ase(self, data: QMDataPoint) -> int:
        """Add to ASE database."""
        atoms = data.atoms.copy()
        atoms.calc = None  # Remove calculator to avoid serialization issues

        kwargs = {
            "energy": data.energy,
            "method": data.method,
            "basis": data.basis,
        }

        if data.forces is not None:
            kwargs["forces"] = data.forces

        idx = self._db.write(atoms, **kwargs, **data.metadata)
        return idx

    def __getitem__(self, idx: int) -> QMDataPoint:
        """Get data point by index.

        Args:
            idx: Index

        Returns:
            QMDataPoint
        """
        if self.backend == "hdf5":
            return self._get_hdf5(idx)
        else:
            return self._get_ase(idx)

    def _get_hdf5(self, idx: int) -> QMDataPoint:
        """Get from HDF5."""
        group = self._file["structures"][str(idx)]

        numbers = group["numbers"][:]
        positions = group["positions"][:]

        atoms = Atoms(numbers=numbers, positions=positions)

        # Load cell if exists
        if "cell" in group:
            atoms.cell = group["cell"][:]
            atoms.pbc = group["pbc"][:]

        energy = float(self._file["energies"][str(idx)][()])

        forces = None
        if str(idx) in self._file["forces"]:
            forces = self._file["forces"][str(idx)][:]

        # Load metadata
        meta_str = self._file["metadata"].attrs.get(str(idx), "{}")
        metadata = json.loads(meta_str)
        method = metadata.pop("method", "")
        basis = metadata.pop("basis", "")

        return QMDataPoint(
            atoms=atoms, energy=energy, forces=forces, method=method, basis=basis, metadata=metadata
        )

    def _get_ase(self, idx: int) -> QMDataPoint:
        """Get from ASE database."""
        row = self._db.get(id=idx)

        atoms = row.toatoms()
        energy = row.get("energy", 0.0)
        forces = row.get("forces")
        method = row.get("method", "")
        basis = row.get("basis", "")

        # Extract other metadata
        metadata = {
            k: v
            for k, v in row.key_value_pairs.items()
            if k not in ("energy", "forces", "method", "basis")
        }

        return QMDataPoint(
            atoms=atoms, energy=energy, forces=forces, method=method, basis=basis, metadata=metadata
        )

    def __len__(self) -> int:
        """Get dataset size."""
        if self.backend == "hdf5":
            return len(self._file["structures"])
        else:
            return len(self._db)

    def __iter__(self) -> Iterator[QMDataPoint]:
        """Iterate over dataset."""
        for i in range(len(self)):
            yield self[i]

    def close(self):
        """Close dataset."""
        if self.backend == "hdf5":
            self._file.close()
        else:
            self._db.connection.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def load_qm9_dataset(path: str | Path, max_molecules: int | None = None) -> Iterator[QMDataPoint]:
    """Load QM9 dataset.

    Args:
        path: Path to QM9 directory or HDF5 file
        max_molecules: Maximum number of molecules to load

    Yields:
        QMDataPoint objects
    """
    from ase.io import read

    path = Path(path)

    if path.is_dir():
        # Load from directory of .xyz files
        files = sorted(path.glob("*.xyz"))
        if max_molecules:
            files = files[:max_molecules]

        for f in files:
            try:
                atoms = read(str(f))
                # Parse energy from comment line if available
                comment = atoms.info.get("comment", "")
                energy = 0.0

                yield QMDataPoint(
                    atoms=atoms,
                    energy=energy,
                    method="DFT",
                    basis="STO-3G",  # QM9 uses this
                    metadata={"source": "QM9", "filename": f.name},
                )
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
    else:
        # Load from HDF5
        with QMLDataset(path, backend="hdf5", mode="r") as dataset:
            for i, dp in enumerate(dataset):
                if max_molecules and i >= max_molecules:
                    break
                yield dp


def split_dataset(
    dataset: QMLDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split dataset into train/val/test.

    Args:
        dataset: QMLDataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(dataset)
    indices = np.arange(n)

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train : n_train + n_val].tolist()
    test_idx = indices[n_train + n_val :].tolist()

    return train_idx, val_idx, test_idx
