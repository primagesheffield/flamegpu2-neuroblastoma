# PRIMAGE Standalone Force Resolution

## Prerequisites

The dependencies below are required for building the project, it's advised to that you use the newest version of any dependencies.

* [CMake](https://cmake.org/) >= 3.12
  * CMake 3.16 is known to have issues on certain platforms
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 9.0
* [git](https://git-scm.com/): Required by CMake for downloading dependencies
* python >= 3
* *Linux:*
  * [make](https://www.gnu.org/software/make/)
  * gcc/g++ >= 6 (version requirements [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements))
      * gcc/g++ >= 7 required for the test suite 
* *Windows:*
  * Visual Studio 2015 or higher (2019 preferred)


## Building

FLAME GPU 2 uses [CMake](https://cmake.org/), as a cross-platform process, for configuring and generating build directives, e.g. `Makefile` or `.vcxproj`. This is used to build the project and any dependencies (e.g. flamegpu2).

Below the most common commands are provided, for the full guide refer to the main [FLAMEGPU2 guide](https://github.com/FLAMEGPU/FLAMEGPU2_dev/blob/master/README.md).

### Linux

Under Linux, `cmake` can be used to generate makefiles specific to your system:

```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SWIG_PYTHON=ON -DBUILD_SWIG_PYTHON_VIRTUALENV=ON -DVISUALISATION=ON
make -j8
```

The option `-j8` enables parallel compilation using upto 8 threads, this is recommended to improve build times.

### Windows

*Note: If installing CMake on Windows ensure CMake is added to the system path, allowing `cmake` to be used via `cmd`, this option is disabled within the installer by default.*

When generating Visual studio project files, using `cmake` (or `cmake-gui`), the platform **must** be specified as `x64`.

Using `cmake` this takes the form `-A x64`:

```
mkdir build && cd build
cmake .. -A x64 -DBUILD_SWIG_PYTHON=ON -DBUILD_SWIG_PYTHON_VIRTUALENV=ON -DVISUALISATION=ON
cmake --build . --config Release
```

## Running
There are two versions of the model implemented in this repository. Using the C++ and Python interfaces to FLAMEGPU2 respectivley. The C++ one is likely to be easier to debug problems with, however most features are supported with the Python interface too (custom transforms/reductions are the only unsupported feature in Python).

The only currently useful runtime argument is `-s` for specifying the number of steps e.g. `-s 10`. Once 10 steps have been completed, the visualisation will remain open (assuming `.join()` is called at the end of `main.cu` or `main.py`). Passing `-s 0` or not providing `-s` will perform unlimited steps until the visualisation is closed.

### C++ model
The built executable can be found inside `build\bin\windows-x64\Release` or `build\bin\linux-x64\Release`.

If you wish to relocate this executable, on Windows you must keep the `.dll` files in the same directory as the executable (or 'install' them to somewhere on the system path), these are required for visualisation.

### Python model
Python does not build the model to a standalone executable, instead it builds flamegpu2 into a python package/wheel, and automatically installs it into a python virtual env. (It's possible to install the package into your main python install, however this has not been used as thoroughly so may have other problems).

If you navigate to the `py_src` directory in a `cmd` or `bash` window, you should be able to call `venv.bat` to activate the virtual environment (on Windows, Linux would need an equivalent `venv.sh` writing). Following this, the model can be called with your regular python command e.g. `python main.py`, and supports the same runtime args as the C++ model (e.g. `python main.py -s 10`).

It may be necessary to install additional packages to the virtual env, e.g. `pip install numpy` whilst the virtual env is active.

In python, to access an flamegpu2 exception's detail `.value.type()` and `.value.what()` can be called on the raised exception object (if handled) to access the exceptions type name and message detail respectively.

## Debugging
The above guide builds Release builds, which execute significantly faster than Debug builds, these cannot be debugged outside of examining the exceptions generated. It may be necessary to instead specify `Debug` at configure/build time (change `Release` to `Debug` in the above build commands and repeat all subsequent commands), at which point the underlying FLAMEGPU2 library can be debugged using your preferred debugger.
