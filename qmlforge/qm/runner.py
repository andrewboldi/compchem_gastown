"""Quantum Chemistry Backend for QMLForge.

This module provides a unified interface for running quantum chemistry calculations
using GPU4PySCF/PySCF with automatic GPU/CPU fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyscf import gto, scf, dft, mp, cc


@dataclass
class QMConfig:
    """Configuration for QM calculations.
    
    Attributes:
        method: QM method (HF, DFT, MP2, CCSD, CCSD(T))
        basis: Basis set name
        xc: Exchange-correlation functional for DFT
        charge: Molecular charge
        spin: Spin multiplicity (2S+1)
        use_gpu: Whether to use GPU acceleration
        device: GPU device ID
        conv_tol: SCF convergence tolerance
        max_cycle: Maximum SCF cycles
        verbose: Verbosity level
    """
    method: Literal["HF", "DFT", "MP2", "CCSD", "CCSD(T)"] = "DFT"
    basis: str = "def2-svp"
    xc: str = "B3LYP"
    charge: int = 0
    spin: int = 1
    use_gpu: bool = True
    device: int = 0
    conv_tol: float = 1e-9
    max_cycle: int = 100
    verbose: int = 3
    
    # Advanced options
    auxbasis: str | None = None  # Auxiliary basis for density fitting
    grids_level: int = 3  # DFT grid level (0-5)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.method == "DFT" and not self.xc:
            raise ValueError("DFT method requires xc functional")


class QMRunner:
    """Quantum chemistry calculation runner with GPU support.
    
    This class provides a high-level interface for running QM calculations
    using GPU4PySCF (GPU) or PySCF (CPU) backends.
    
    Example:
        >>> from ase import Atoms
        >>> water = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        >>> qm = QMRunner(method='DFT', basis='def2-svp', xc='B3LYP')
        >>> energy = qm.calculate_energy(water)
        >>> forces = qm.calculate_forces(water)
    """
    
    def __init__(self, config: QMConfig | None = None, **kwargs):
        """Initialize QM runner.
        
        Args:
            config: QMConfig object or None (uses defaults)
            **kwargs: Override config attributes
        """
        if config is None:
            config = QMConfig()
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config option: {key}")
        
        self.config = config
        self._gpu_available = self._check_gpu()
        self._last_calc = None
        
        logger.info(f"QMRunner initialized: {config.method}/{config.basis}")
        logger.info(f"GPU available: {self._gpu_available}")
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available and working."""
        if not self.config.use_gpu:
            return False
        
        try:
            import cupy as cp
            cp.cuda.Device(self.config.device).use()
            # Try a simple operation
            x = cp.array([1.0, 2.0, 3.0])
            del x
            logger.info(f"GPU {self.config.device} is available")
            return True
        except ImportError:
            logger.warning("CuPy not installed, GPU acceleration unavailable")
            return False
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False
    
    def _ase_to_pyscf(self, atoms: Atoms) -> "gto.Mole":
        """Convert ASE Atoms to PySCF Mole object.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            PySCF Mole object
        """
        from pyscf import gto
        
        mol = gto.Mole()
        mol.atom = [[atom.symbol, atom.position] for atom in atoms]
        mol.basis = self.config.basis
        mol.charge = self.config.charge
        mol.spin = self.config.spin - 1  # PySCF uses 2S, not 2S+1
        mol.verbose = self.config.verbose
        
        if self.config.auxbasis:
            mol.auxbasis = self.config.auxbasis
        
        mol.build()
        return mol
    
    def _create_calc(self, mol: "gto.Mole") -> Any:
        """Create PySCF calculator based on method.
        
        Args:
            mol: PySCF Mole object
            
        Returns:
            PySCF calculator object
        """
        if self._gpu_available:
            logger.info("Using GPU acceleration")
            return self._create_gpu_calc(mol)
        else:
            logger.info("Using CPU")
            return self._create_cpu_calc(mol)
    
    def _create_cpu_calc(self, mol: "gto.Mole") -> Any:
        """Create CPU-based PySCF calculator."""
        method = self.config.method
        
        if method == "HF":
            from pyscf import scf
            calc = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        elif method == "DFT":
            from pyscf import dft
            calc = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            calc.xc = self.config.xc
            calc.grids.level = self.config.grids_level
        elif method == "MP2":
            from pyscf import scf, mp
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
            calc = mp.MP2(mf)
        elif method in ("CCSD", "CCSD(T)"):
            from pyscf import scf, cc
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
            calc = cc.CCSD(mf)
            if method == "CCSD(T)":
                # Store for later
                calc._run_triples = True
        else:
            raise ValueError(f"Unknown method: {method}")
        
        calc.conv_tol = self.config.conv_tol
        calc.max_cycle = self.config.max_cycle
        return calc
    
    def _create_gpu_calc(self, mol: "gto.Mole") -> Any:
        """Create GPU-accelerated calculator using GPU4PySCF."""
        import gpu4pyscf
        from gpu4pyscf import scf, dft
        
        method = self.config.method
        
        if method == "HF":
            calc = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        elif method == "DFT":
            calc = dft.rks.RKS(mol) if mol.spin == 0 else dft.uks.UKS(mol)
            calc.xc = self.config.xc
            calc.grids.level = self.config.grids_level
        elif method == "MP2":
            # GPU MP2 via PySCF with GPU SCF
            from pyscf import mp
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
            mf.kernel()
            calc = mp.MP2(mf)
        else:
            logger.warning(f"{method} not GPU-optimized, falling back to CPU")
            return self._create_cpu_calc(mol)
        
        calc.conv_tol = self.config.conv_tol
        calc.max_cycle = self.config.max_cycle
        return calc
    
    def calculate_energy(self, atoms: Atoms) -> float:
        """Calculate total energy.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Total energy in eV
        """
        mol = self._ase_to_pyscf(atoms)
        calc = self._create_calc(mol)
        
        logger.info(f"Running {self.config.method} energy calculation...")
        
        if self.config.method in ("MP2", "CCSD"):
            calc.kernel()
            energy = calc.e_tot if hasattr(calc, 'e_tot') else calc.e_hf + calc.e_corr
        elif self.config.method == "CCSD(T)":
            calc.kernel()
            energy = calc.e_tot
            if hasattr(calc, '_run_triples') and calc._run_triples:
                from pyscf.cc import ccsd_t
                e_t = ccsd_t.kernel(calc, calc.ao_repr=True)
                energy += e_t
        else:
            energy = calc.kernel()
        
        # Convert from Hartree to eV
        energy_ev = energy * Hartree
        
        logger.info(f"Energy: {energy_ev:.6f} eV")
        self._last_calc = calc
        
        return energy_ev
    
    def calculate_forces(self, atoms: Atoms) -> np.ndarray:
        """Calculate atomic forces.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            Forces array (N_atoms, 3) in eV/Å
        """
        mol = self._ase_to_pyscf(atoms)
        calc = self._create_calc(mol)
        
        logger.info(f"Running {self.config.method} force calculation...")
        
        # Run SCF first
        calc.kernel()
        
        # Calculate gradients
        if self._gpu_available and self.config.method in ("HF", "DFT"):
            # GPU-accelerated gradients
            if hasattr(calc, 'nuc_grad_method'):
                grad_calc = calc.nuc_grad_method()
                grad = grad_calc.kernel()
            else:
                # Fallback to numeric gradients on CPU
                grad = self._numeric_gradients(atoms)
        else:
            # CPU gradients
            if hasattr(calc, 'nuc_grad_method'):
                grad_calc = calc.nuc_grad_method()
                grad = grad_calc.kernel()
            else:
                grad = self._numeric_gradients(atoms)
        
        # Convert from Hartree/Bohr to eV/Å
        # Force = -gradient
        forces = -grad * Hartree / Bohr
        
        logger.info(f"Max force: {np.max(np.abs(forces)):.6f} eV/Å")
        self._last_calc = calc
        
        return forces
    
    def _numeric_gradients(self, atoms: Atoms, delta: float = 0.001) -> np.ndarray:
        """Calculate numerical gradients as fallback.
        
        Args:
            atoms: ASE Atoms object
            delta: Displacement in Å
            
        Returns:
            Gradient array (N_atoms, 3) in Hartree/Å
        """
        logger.warning("Using numerical gradients (slow)")
        
        gradients = np.zeros((len(atoms), 3))
        energy0 = self.calculate_energy(atoms)
        
        for i in range(len(atoms)):
            for j in range(3):
                # Forward displacement
                atoms_plus = atoms.copy()
                atoms_plus.positions[i, j] += delta
                energy_plus = self.calculate_energy(atoms_plus)
                
                # Backward displacement
                atoms_minus = atoms.copy()
                atoms_minus.positions[i, j] -= delta
                energy_minus = self.calculate_energy(atoms_minus)
                
                # Central difference
                gradients[i, j] = (energy_plus - energy_minus) / (2 * delta)
        
        # Convert from eV/Å to Hartree/Å
        gradients /= Hartree
        
        return gradients
    
    def optimize_geometry(
        self, 
        atoms: Atoms, 
        fmax: float = 0.05, 
        steps: int = 200,
        log_interval: int = 10
    ) -> tuple[Atoms, bool]:
        """Optimize molecular geometry.
        
        Args:
            atoms: Initial ASE Atoms object
            fmax: Force convergence criterion (eV/Å)
            steps: Maximum optimization steps
            log_interval: Logging frequency
            
        Returns:
            Tuple of (optimized_atoms, converged)
        """
        from ase.optimize import LBFGS
        
        logger.info(f"Starting geometry optimization (fmax={fmax} eV/Å)...")
        
        # Create ASE calculator wrapper
        calc = QMLForgeCalculator(self)
        atoms.calc = calc
        
        # Run optimization
        opt = LBFGS(atoms, logfile=None)
        
        converged = False
        for i in range(0, steps, log_interval):
            opt.run(fmax=fmax, steps=log_interval)
            
            if opt.converged():
                converged = True
                logger.info(f"Optimization converged in {i + len(opt)} steps")
                break
            
            if i % log_interval == 0:
                logger.info(f"Step {i}: E={atoms.get_potential_energy():.6f} eV")
        
        if not converged:
            logger.warning(f"Optimization did not converge in {steps} steps")
        
        return atoms, converged


class QMLForgeCalculator(Calculator):
    """ASE Calculator wrapper for QMLForge QM backend.
    
    This allows using QMRunner with ASE's optimization and dynamics tools.
    
    Example:
        >>> from ase import Atoms
        >>> from ase.optimize import BFGS
        >>> water = Atoms('H2O', ...)
        >>> qm = QMRunner(method='DFT', basis='def2-svp', xc='B3LYP')
        >>> water.calc = QMLForgeCalculator(qm)
        >>> opt = BFGS(water)
        >>> opt.run(fmax=0.05)
    """
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, qm_runner: QMRunner, **kwargs):
        """Initialize calculator.
        
        Args:
            qm_runner: QMRunner instance
            **kwargs: ASE Calculator kwargs
        """
        super().__init__(**kwargs)
        self.qm_runner = qm_runner
    
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """Calculate properties."""
        if atoms is not None:
            self.atoms = atoms.copy()
        
        if properties is None:
            properties = self.implemented_properties
        
        if system_changes is None:
            system_changes = all_changes
        
        # Calculate energy and forces
        self.results['energy'] = self.qm_runner.calculate_energy(self.atoms)
        self.results['forces'] = self.qm_runner.calculate_forces(self.atoms)


def benchmark_qm(atoms: Atoms, config: QMConfig | None = None) -> dict[str, Any]:
    """Benchmark QM calculation performance.
    
    Args:
        atoms: ASE Atoms object
        config: QM configuration
        
    Returns:
        Dictionary with timing and memory information
    """
    import time
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    qm = QMRunner(config)
    
    # Benchmark energy
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    t0 = time.time()
    energy = qm.calculate_energy(atoms)
    t_energy = time.time() - t0
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Benchmark forces
    t0 = time.time()
    forces = qm.calculate_forces(atoms)
    t_forces = time.time() - t0
    
    return {
        'energy': energy,
        'forces': forces,
        'time_energy': t_energy,
        'time_forces': t_forces,
        'memory_used_mb': mem_after - mem_before,
        'gpu_used': qm._gpu_available,
        'n_atoms': len(atoms),
        'method': qm.config.method,
        'basis': qm.config.basis,
    }
