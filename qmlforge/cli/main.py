"""QMLForge CLI - Command-line interface."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()

# Main Typer app
app = typer.Typer(
    name="qmlforge",
    help="QMLForge - Quantum Machine Learning Framework for Computational Chemistry",
    rich_markup_mode="rich",
    add_completion=False,
)

# Subcommand groups
qm_app = typer.Typer(help="Quantum chemistry calculations")
data_app = typer.Typer(help="Dataset management")
train_app = typer.Typer(help="MLIP training")
sim_app = typer.Typer(help="Molecular dynamics simulations")

app.add_typer(qm_app, name="qm")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(sim_app, name="sim")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
):
    """QMLForge CLI - GPU-optimized quantum ML for chemistry."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)


@qm_app.command("energy")
def qm_energy(
    molecule: Path = typer.Argument(..., help="Path to molecule file (XYZ, PDB, etc.)"),
    method: str = typer.Option("DFT", "--method", "-m", help="QM method (HF, DFT, MP2, CCSD)"),
    basis: str = typer.Option("def2-svp", "--basis", "-b", help="Basis set"),
    xc: str = typer.Option("B3LYP", "--xc", "-x", help="DFT functional"),
    charge: int = typer.Option(0, "--charge", "-c", help="Molecular charge"),
    spin: int = typer.Option(1, "--spin", "-s", help="Spin multiplicity"),
    gpu: bool = typer.Option(True, "--gpu/--cpu", help="Use GPU acceleration"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Calculate single-point energy."""
    try:
        from ase.io import read
        from qmlforge.qm import QMRunner, QMConfig

        console.print(
            Panel.fit(
                f"[bold blue]QMLForge QM Energy Calculation[/bold blue]\n"
                f"Method: {method}/{basis}\n"
                f"Molecule: {molecule}"
            )
        )

        # Load molecule
        atoms = read(str(molecule))
        console.print(f"Loaded: {len(atoms)} atoms")

        # Setup QM
        config = QMConfig(
            method=method,
            basis=basis,
            xc=xc,
            charge=charge,
            spin=spin,
            use_gpu=gpu,
        )

        qm = QMRunner(config)

        # Calculate energy
        with console.status("[bold green]Running QM calculation..."):
            energy = qm.calculate_energy(atoms)

        console.print(f"\n[bold green]Energy: {energy:.6f} eV[/bold green]")

        if output:
            with open(output, "w") as f:
                f.write(f"Energy: {energy:.6f} eV\n")
            console.print(f"Saved to: {output}")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@qm_app.command("forces")
def qm_forces(
    molecule: Path = typer.Argument(..., help="Path to molecule file"),
    method: str = typer.Option("DFT", "--method", "-m", help="QM method"),
    basis: str = typer.Option("def2-svp", "--basis", "-b", help="Basis set"),
    xc: str = typer.Option("B3LYP", "--xc", "-x", help="DFT functional"),
    gpu: bool = typer.Option(True, "--gpu/--cpu", help="Use GPU acceleration"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Calculate atomic forces."""
    try:
        from ase.io import read
        from qmlforge.qm import QMRunner, QMConfig
        import numpy as np

        console.print(
            Panel.fit(
                f"[bold blue]QMLForge QM Forces Calculation[/bold blue]\nMethod: {method}/{basis}"
            )
        )

        atoms = read(str(molecule))
        config = QMConfig(method=method, basis=basis, xc=xc, use_gpu=gpu)
        qm = QMRunner(config)

        with console.status("[bold green]Calculating forces..."):
            forces = qm.calculate_forces(atoms)

        console.print("\n[bold]Forces (eV/Å):[/bold]")
        for i, (atom, force) in enumerate(zip(atoms, forces)):
            console.print(
                f"  {i:3d} {atom.symbol:2s}  {force[0]:10.6f}  {force[1]:10.6f}  {force[2]:10.6f}"
            )

        if output:
            np.savetxt(output, forces, header="Forces (eV/Å)")
            console.print(f"\nSaved to: {output}")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@qm_app.command("optimize")
def qm_optimize(
    molecule: Path = typer.Argument(..., help="Path to molecule file"),
    method: str = typer.Option("DFT", "--method", "-m", help="QM method"),
    basis: str = typer.Option("def2-svp", "--basis", "-b", help="Basis set"),
    xc: str = typer.Option("B3LYP", "--xc", "-x", help="DFT functional"),
    fmax: float = typer.Option(0.05, "--fmax", help="Force convergence criterion (eV/Å)"),
    steps: int = typer.Option(200, "--steps", "-n", help="Maximum optimization steps"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output geometry file"),
):
    """Optimize molecular geometry."""
    try:
        from ase.io import read, write
        from qmlforge.qm import QMRunner, QMConfig

        console.print(
            Panel.fit(
                f"[bold blue]QMLForge Geometry Optimization[/bold blue]\n"
                f"Method: {method}/{basis}\n"
                f"fmax: {fmax} eV/Å"
            )
        )

        atoms = read(str(molecule))
        config = QMConfig(method=method, basis=basis, xc=xc)
        qm = QMRunner(config)

        opt_atoms, converged = qm.optimize_geometry(atoms, fmax=fmax, steps=steps)

        status = (
            "[bold green]Converged[/bold green]"
            if converged
            else "[bold yellow]Not converged[/bold yellow]"
        )
        console.print(f"\nOptimization: {status}")
        console.print(f"Final energy: {opt_atoms.get_potential_energy():.6f} eV")

        if output:
            write(output, opt_atoms)
            console.print(f"Saved optimized geometry to: {output}")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@data_app.command("convert")
def data_convert(
    input_file: Path = typer.Argument(..., help="Input file"),
    output_file: Path = typer.Argument(..., help="Output file"),
    format_in: Optional[str] = typer.Option(
        None, "--from", help="Input format (auto-detect if not specified)"
    ),
    format_out: Optional[str] = typer.Option(
        None, "--to", help="Output format (auto-detect if not specified)"
    ),
):
    """Convert between molecular file formats."""
    try:
        from ase.io import read, write

        atoms = read(str(input_file), format=format_in)
        write(str(output_file), atoms, format=format_out)

        console.print(f"[bold green]Converted:[/bold green] {input_file} -> {output_file}")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("version")
def version():
    """Show version information."""
    from qmlforge import __version__

    console.print(
        Panel.fit(
            f"[bold blue]QMLForge[/bold blue]\n"
            f"Version: {__version__}\n"
            f"Python: {sys.version.split()[0]}"
        )
    )


if __name__ == "__main__":
    app()
