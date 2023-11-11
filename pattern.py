import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import os
from typing import Tuple
import click

# Constants
FILE_PATH = "pattern"


def reaction_diffusion(
    a: np.ndarray,
    b: np.ndarray,
    da: float,
    db: float,
    feed: float,
    k: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs the reaction-diffusion process.

    Args:
      a: The 'A' component of the system.
      b: The 'B' component of the system.
      da: Diffusion rate of 'A'.
      db: Diffusion rate of 'B'.
      feed: Feed rate.
      k: Kill rate.
      steps: Number of simulation steps.

    Returns:
      Tuple of numpy arrays representing the final state of 'A' and 'B'.
    """
    kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])

    for _ in range(steps):
        la = convolve(a, kernel, mode="wrap")
        lb = convolve(b, kernel, mode="wrap")
        ab2 = a * b**2
        a += da * la - ab2 + feed * (1 - a)
        b += db * lb + ab2 - (k + feed) * b
        np.clip(a, 0, 1, out=a)  # Ensuring values are within [0, 1]
        np.clip(b, 0, 1, out=b)

    return a, b


def interactive_simulation_steps(
    steps: int,
    da: float = 1.0,
    db: float = 0.5,
    feed: float = 0.055,
    k: float = 0.062,
    colormap: str = "magma",
) -> plt.Figure:
    """Runs the interactive simulation for a given number of steps.

    Args:
      steps: Number of simulation steps.
      da, db, feed, k: Parameters for the reaction-diffusion process.
      colormap: Colormap for the resulting image.

    Returns:
      Matplotlib figure object.
    """
    size = 100
    a = np.ones((size, size))
    b = np.zeros((size, size))
    np.random.seed(42)
    b[
        size // 2 - 10 : size // 2 + 10, size // 2 - 10 : size // 2 + 10
    ] = np.random.rand(20, 20)
    a, b = reaction_diffusion(a, b, da, db, feed, k, steps)

    fig = plt.figure()
    plt.imshow(b, cmap=colormap, interpolation="bilinear")
    plt.axis("off")
    return fig


def steps_generator(
    min_step: int,
    max_step: int,
    interval: int,
    a: float,
    b: float,
    feed: float,
    kill: float,
):
    """Generates images for a range of steps at specified intervals.

    Args:
      min_step: Minimum number of steps.
      max_step: Maximum number of steps.
      interval: Interval between steps.
    """

    folder_name = f"{min_step}_{max_step}_{interval}_{a}_{b}_{feed}_{kill}_pattern"
    directory_path = os.path.join(FILE_PATH, folder_name)
    os.makedirs(directory_path, exist_ok=True)

    print(f"Generating pattern to {directory_path} ...")

    for i in range(min_step, max_step, interval):
        print(f"image {i} generating ...")
        img = interactive_simulation_steps(i, a, b, feed, kill)
        file_name = os.path.join(directory_path, f"{i}.png")
        img.savefig(file_name)
        print(f"image {i} saved ...")


@click.command()
@click.option(
    "--min_step",
    default=5000,
    type=int,
    help="Minimum number of steps.",
)
@click.option(
    "--max_step",
    default=6000,
    type=int,
    help="Maximum number of steps.",
)
@click.option(
    "--interval",
    default=500,
    type=int,
    help="Interval between steps.",
)
@click.option(
    "--a",
    default=1.00,
    type=float,
    help="a",
)
@click.option(
    "--b",
    default=0.3,
    type=float,
    help="b",
)
@click.option(
    "--feed",
    default=0.07,
    type=float,
    help="feed",
)
@click.option(
    "--kill",
    default=0.07,
    type=float,
    help="kill",
)
def main(
    min_step: int,
    max_step: int,
    interval: int,
    a: float,
    b: float,
    feed: float,
    kill: float,
):
    """Command-line tool to generate reaction-diffusion pattern images.

    Args:
      min_step: Minimum number of steps for the simulation.
      max_step: Maximum number of steps for the simulation.
      interval: Interval between steps for generating images.
    """
    steps_generator(min_step, max_step, interval, a, b, feed, kill)


if __name__ == "__main__":
    main()
