
# TensorFlow GPU Setup Guide on WSL (Windows Subsystem for Linux)

## Notes
- 10/05/2024, the perch repo updated at some point from tf v2.13.0 to tf v2.15.0. The cuda software versions are now incompatible and the gpu can't be found. This doc needs updating to the versions which work with tf 2.15.0
- The cuda toolkit and cuDNN versions were set to match TF v2.13.0 for Perch system wide. This might not be compatible with future projects. However, anaconda seems to be able to set these version within it and still work, at least if installing the latest version of TF for WSL from: https://www.tensorflow.org/install/pip

## Prerequisites
- **Windows 10/11**: This guide is based on Windows 11, but the steps should be similar for Windows 10.
- **GPU Setup on Windows**: Ensure you have the NVIDIA driver and toolkit installed on Windows.
    - Perform checks to see if TensorFlow can find the GPU on Windows. 
    - Example: Create a quick Anaconda environment with TensorFlow and check if the GPU can be found.
- **Install WSL2**: Using Ubuntu 22.04. Follow the official guide to install WSL2.
- **(Optional) Install Miniconda**: Not necessary for Perch but useful for other projects.

## Install Other Prerequisites
On a terminal in WSL, run the following to install ffmpeg and libsndfile1, which are dependencies for Perch:
```bash
sudo apt install ffmpeg libsndfile1
```

## Steps to Install Perch
1. **Install Poetry**:
    Open a PowerShell terminal in VS Code, and run:
    ```bash
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```
    Check Poetry installation:
    ```bash
    poetry --version
    ```

2. **Clone Perch Repo**:
    Navigate to the folder where you want to clone Perch, and run:
    ```bash
    git clone https://github.com/google-research/perch
    ```

3. **Install Perch**:
    Navigate to the Perch repository folder and install:
    ```bash
    cd perch
    poetry install
    ```
    Activate the poetry environment:
    ```bash
    poetry shell
    ```
    Check if GPU works. It likely won'twork at this stage, but if it does, you will see output similar to `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`:
    ```bash
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

4. **Test Notebooks**:
    Verify that Jupyter and ipykernel are present (they should be if the above installation was successful):
    ```bash
    poetry show
    ```
    For the first time, set up a kernel for Jupyter:
    ```bash
    poetry run python -m ipykernel install --user --name=perch-kernel-v1 --display-name="perch-kernel-v1"
    ```
    Subsequently, you can simply run:
    ```bash
    poetry run jupyter notebook
    ```
    Or, open a Jupyter notebook `.ipynb` file and select the `perch-kernel-v1` kernel to test.

5. **Check GPU Access in WSL**:
    Check GPU visibility in WSL:
    ```bash
    nvidia-smi
    ```
    Check if NVCC is installed. It most likely won't be, but if present it must show version 11.8, otherwise it must be uninstalled and reinstalled:
    ```bash
    nvcc -V
    ```

6. **Install NVIDIA Toolkit and cuDNN**:
    - **CUDA Toolkit Installation**:
        1. First, check if nvcc is installed, if it is skip to cuDNN Installation:
            ```
            nvcc --version
            ```
           This will display the CUDA compiler version, confirming it is installed. If not installed continue below.
        2. Navigate to the [CUDA Toolkit 11.8 Download Page](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) and follow the instructions for WSL Ubuntu.
        3. Run the following commands to download and install the CUDA Toolkit:
            ```
            wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
            sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
            wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
            sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
            sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/*.pub /usr/share/keyrings/
            sudo apt-get update
            sudo apt-get -y install cuda-11-8
            ```
        4. Update your `PATH` to include CUDA binaries:
            ```
            export PATH=/usr/local/cuda/bin:$PATH
            source ~/.bashrc
            ```
           Then, edit `~/.bashrc` to permanently add the path:
            ```
            nano ~/.bashrc
            # Append the following line at the end:
            export PATH=/usr/local/cuda/bin:$PATH
            ```
           Write and exit, then apply these changes system wide with `source ~/.bashrc`.

        5. Verify CUDA installation:
            ```
            nvcc --version
            ```
           This should display the CUDA compiler version, confirming a successful installation.

    - **cuDNN Installation**:
        1. Download the cuDNN package for Ubuntu 22.04 from the [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive). Select: `Download cuDNN v8.6.0 (October 3rd, 2022), for CUDA 11.x` and then `Local Installer for Ubuntu22.04 x86_64 (Deb)`. Make sure this gets saved to the downloads folder in step 2 below (otherwise navigate to where it was saved instead).
        2. Install the downloaded package using `dpkg`:
            ```
            cd /mnt/c/Users/<YourUsername>/Downloads
            sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
            ```
        3. Add the GPG key for the cuDNN repository:
            ```
            sudo cp /var/cudnn-local-repo-ubuntu2204-8.6.0.163/cudnn-local-FAED14DD-keyring.gpg /usr/share/keyrings/
            ```
        4. Update packages and install cuDNN:
            ```
            sudo apt-get update
            apt-cache search cudnn
            ```
           Find the cuDNN package that matches your CUDA version and install it.

        5. Verify TensorFlow can access the GPU:
            ```
            python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
            ```
           This command should list the available GPU devices if cuDNN and CUDA are correctly installed and configured.
``` &#8203;``【oaicite:0】``&#8203;


## Verifying TensorFlow GPU Access
After setting up CUDA and cuDNN, verify TensorFlow can access the GPU:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Resolving CUDA and cuDNN Library Path Issues

7. **Fixing Library Path for GPU Access in TensorFlow**:
    - Even after successful installation, and success with test TF scripts, the embedding script might throw an error at setup indicating it cannot find certain CUDA or cuDNN libraries (e.g., `libcuda.so` or `libcudnn_cnn_infer.so.8`). This is typically due to these libraries not being in the `LD_LIBRARY_PATH`.
    
    - **Identify Library Locations**:
        Use the `find` command to locate the missing libraries within your WSL environment. For `libcuda.so`, run:
        ```bash
        sudo find / -name libcuda.so 2>/dev/null
        ```
        This command may reveal multiple paths. For WSL environments, the relevant path usually includes `/usr/lib/wsl/lib`.

    - **Update `.bashrc` to Include Library Paths**:
        Edit `.bashrc` file to include the paths to the CUDA and cuDNN libraries, ensuring TensorFlow can access them:
        ```bash
        nano ~/.bashrc
        # Add the following line to the end of the file
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
        ```
        After adding the line, save and exit the editor (in `nano`, use `Ctrl+O`, `Enter` to save and `Ctrl+X` to exit).

    - **Apply the Changes**:
        To make the changes effective, source `.bashrc` or open a new terminal window:
        ```bash
        source ~/.bashrc
        ```

    - **Verify the Setup**:
        Test again to ensure TensorFlow can now access the GPU without errors. Run a simple Python command to list the available GPU devices:
        ```bash
        python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
        ```
        Also rerun the embed script and check the model setup can now complete.
        If correctly configured, TensorFlow should now be able to utilize the GPU for computations.

**Note**: It's crucial to ensure the correct paths are added to `LD_LIBRARY_PATH` to avoid potential conflicts between different CUDA or cuDNN versions installed on your system.