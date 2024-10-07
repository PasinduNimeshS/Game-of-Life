Game-of-Life

This repository contains a PyCUDA implementation of Game of Life, executed in Google Colab. The project demonstrates how to use CUDA to accelerate the cellular automaton simulation on a GPU.

Overview :-

Conway's Game of Life is a zero-player game where the evolution of the grid is determined by its initial state, requiring no further input. The game follows these simple rules:

•Any live cell with fewer than two live neighbors dies (underpopulation).

•Any live cell with two or three live neighbors lives on to the next generation.

•Any live cell with more than three live neighbors dies (overpopulation).

•Any dead cell with exactly three live neighbors becomes a live cell (reproduction).

This implementation uses PyCUDA to accelerate the computations on a GPU, visualizing the results using Matplotlib.

Setup :-

To run the code, you will need access to a machine with CUDA-enabled hardware and the required dependencies.

Steps :-

1.Check GPU Availability
!nvidia-smi
Ensure that your system has a CUDA-capable GPU.

2.Install Dependencies
!pip install numpy matplotlib

3.Write CUDA Kernel for the Game of Life The CUDA code is written in C++ and defines the update function, which handles the grid evolution based on the game's rules. The code is saved into a .cu file.
cuda_code = """
extern "C"
__global__ void update(int *current, int *next, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    ...
}
"""
with open('game_of_life.cu', 'w') as f:
    f.write(cuda_code)

4.Compile the CUDA Kernel
!nvcc game_of_life.cu -o game_of_life

5.Install PyCUDA
!pip install pycuda

6.Run the Simulation After setting up the kernel and memory allocation on the GPU, the game runs in a loop where each generation of the grid is computed and displayed using Matplotlib.
// Initialize parameters
width, height = 100, 100  # Dimensions of the grid
...
The simulation will run until interrupted by the user.

How It Works

1.Grid Initialization: The grid is initialized randomly, with each cell either alive (1) or dead (0).

2.CUDA Kernel: The CUDA kernel computes the next state of each cell by counting its alive neighbors and applying the rules of the Game of Life.

3.Visualization: The state of the grid is displayed using Matplotlib at each iteration, giving a visual representation of the game in action.

Requirements :-
•Python 3.x

•PyCUDA

•NumPy

•Matplotlib

•NVIDIA GPU with CUDA support

