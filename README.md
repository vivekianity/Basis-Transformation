# Quantum Circuit Basis Obfuscation

This project implements a **quantum circuit obfuscation** technique using random basis transformations (via `U3` gates).  
It takes an input QASM circuit, applies obfuscation, executes both the original and obfuscated circuits,  
and compares their results using semantic accuracy, TVD (Total Variation Distance), and DFC (Degree of Functional Corruption).

## Features
- Parse **QASM 2.0** and **QASM 3.0** circuits.
- Automatically adds measurement registers if missing.
- Applies **random U3 basis transformations** for obfuscation.
- Executes circuits on **Qiskit AerSimulator**.
- Compares results between original and obfuscated circuits.
- Visualizes both circuits and result distributions using `matplotlib`.

## Requirements

1. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

2. Modify the variable `file_path` in `main.py` to point to the QASM file of the quantum circuit.
3. Run `main.py`

## Acknowledgement

If you use this repository or build upon this work, please cite our paper:

```bibtex
@article{basis_transformation_2025,
  title={Protecting Quantum Circuits Through Compiler-Resistant Obfuscation},
  author={Parayil, Pradyun and Raj, Amal and Balachandran, Vivek},
  year={2025},
  eprint={2512.19314},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  doi={10.48550/arXiv.2512.19314},
  url={https://arxiv.org/abs/2512.19314}
}