# CRLQAS-code

# Installation
Start by verifying (installing, or loading modules on HPC cluster or cloud platform) the following software versions & libraries for CUDA (11.8.0), cuDNN (8.6), Python (3.10), and GCC (12.2.0).

```
Python/3.10.8-GCCcore-12.2.0
cuDNN/8.6.0.163-CUDA-11.8.0
```

# Create a Python Virtual Environment

```
python3 -m venv --upgrade-deps /path/to/new/virtual/environment
```

# Activate Your New Virtual Environment

```
source activate /path/to/new/virtual/environment
```

# Install All the Required Packages Given In This Repository

```
pip install -r crlqas_requirements.txt
```

# The Structure of the repository

Similar to the article, the code in this repository is divided into **noiseless** and **noisy** scenarios. The **noiseless** code (in the **qulacs_noiseless** folder) operates purely on an open-source quantum simulator [**Qulacs**](https://github.com/qulacs/qulacs). Meanwhile, the **noisy** (in the **jax_noisy** folder) code is built upon the [**JAX**](https://github.com/google/jax), which enables high-performance numerical computing on GPU.

The **jax_noisy** folder contains all the **main_*.py** files that run different noisy settings for different qubit system sizes. It also contains a folder named **environment** with the codes for the **RL environments** for all the noise settings and qubit system. The **main_*.py** files utilize the **RL environments** from the **environment** folder. This (**jax_noisy**) folder contains the **PTM_files** folder, which are the Pauli-Transfer matrices of the noisy quantum gates we computed offline. We later utilized these matrices in the code.

Meanwhile, the **environment** folder contains the **environment_*.py** files and the **VQE_files** folder with **VQE_*.py** files (these are the codes for the variational quantum eigensolvers on different noisy scenarios). The **environment_*.py** utilizes these **VQE_*.py** from the folder to get energy per step of an episode, which serves as an intermittent signal for the **RL agent**.

Finally, the **VQE** folder contains the **VQE_*.py** files and the **quant** folder with **quant_*.py** files (these are the linear algebra libraries for noisy quantum circuit simulation on different scenarios). The **VQE_*.py** utilizes this **quant_*.py** from the folder to get the respective **PTM_files** for different noise models and returns the expectation value.


# Running CRLQAS on a noisy and noiseless scenario

Here, you need to set the "**wandb_project**" variables in your **main_*.py** into the [**WanDB**](https://wandb.ai/) projects you created, and similarly "**wandb_entity**" objects to the entities you have in WanDB. To turn on the **WanDB**, one can change. 

```
os.environ['WANDB_DISABLED'] = "False"
```

Upon changing your working directory to the **jax_noisy** folder for **noisy** and to the **qulacs_noiseless** folder for the **noiseless** experiments, you run experiments following the prompt convention below:

```
python main_*.py --seed an_integer --config name_of_the_config_file_without_cfg_extension --experiment_name "name_of_the_folder_containing_the_config_files/"
```

For example, in order **to run the 2 qubit noisy VQE problem for the $H_2$ molecule** using the device noise model of **_IBM Mumbai_** device with **median noise** parameters, you first need to change to the **jax_noisy** folder and run the following command:

```
python main_2q_median.py --seed 123 --config lbmt_adam1_H22q0p7414_1e3_ibmq_mumbai_median --experiment_name "finalize/"
```

Similarly, **to run a 6 qubit noiseless VQE problem** for the **$LiH$ molecule** using the Qulacs simulator, you need to switch to **qulacs_noiseless** folder and run the following command:

```
python main.py --seed 123 --config lbmt_cobyla_LiH6q2p2_rand_halt --experiment_name "finalize/"
```

To compare the CRLQAS for 4 qubit **$H_2$ molecule** with QCAS [**Du et al. 2022**](https://www.nature.com/articles/s41534-022-00570-y), you need to run the following command:

```
python main_qas.py --seed 123 --config lbmt_cobyla_randhalt_qas --experiment_name "finalize/"
```
All the simulations reported in our paper use 100-102 seeds for 2-4 and 8 qubit simulations. Meanwhile, for 6 qubits, we use 0-4 seeds.







