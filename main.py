import sys, time, random, math
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import AerSimulator
import qiskit.qasm3, qiskit.qasm2
from qiskit.circuit.library import U3Gate, UnitaryGate
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Operator

def add_measurements(circuit):
    has_measurements = any(instruction.operation.name == "measure" for instruction in circuit.data)
    if not has_measurements:
        if not circuit.clbits:
            print("Warning: No classical registers found. Adding one.")
            circuit.add_register(ClassicalRegister(len(circuit.qubits), 'c'))
        
        num_clbits = len(circuit.clbits)
        qubits_to_measure = circuit.qubits[:min(num_clbits, len(circuit.qubits))]
        
        for i, qubit in enumerate(qubits_to_measure):
            circuit.measure(qubit, circuit.clbits[i])
        print(f"Added measurements for {len(qubits_to_measure)} qubits to classical register.")
    
    return circuit

def generate_random_u3_params():
    theta = random.uniform(0, math.pi)
    phi = random.uniform(0, 2 * math.pi)
    lam = random.uniform(0, 2 * math.pi)
    return theta, phi, lam

def get_basis_block(num_qubits, u3_params, inverse=False, label="Basis"):
    theta, phi, lam = u3_params
    block = QuantumCircuit(num_qubits, name=label)
    gate = U3Gate(-theta, -lam, -phi) if inverse else U3Gate(theta, phi, lam)
    for i in range(num_qubits):
        block.append(gate, [i])
    return block  

def create_transformed_gate(gate, u3_params, block_id):
    gate_name_map = {
        'h': 'H', 'cx': 'CNOT', 'x': 'X', 'y': 'Y', 'z': 'Z',
        't': 'T', 's': 'S', 'rx': 'RX', 'ry': 'RY', 'rz': 'RZ',
        'cu': 'CU', 'cp': 'CP', 'cry': 'CRY', 'swap': 'SWAP'
    }
    gate_name = gate_name_map.get(gate.name.lower(), gate.name.upper())
    num_qubits = gate.num_qubits
    
    theta, phi, lam = u3_params
    U_single = U3Gate(theta, phi, lam)
    U_inv_single = U3Gate(-theta, -lam, -phi)
    
    U_single_matrix = Operator(U_single).data
    U_inv_single_matrix = Operator(U_inv_single).data
    
    U_matrix = U_single_matrix
    U_inv_matrix = U_inv_single_matrix
    for _ in range(num_qubits - 1):
        U_matrix = np.kron(U_matrix, U_single_matrix)
        U_inv_matrix = np.kron(U_inv_matrix, U_inv_single_matrix)
    
    gate_matrix = Operator(gate).data
    combined_matrix = U_matrix @ gate_matrix @ U_inv_matrix
    
    transformed_gate = UnitaryGate(
        combined_matrix,
        label=f"Obf_{gate_name}_{block_id}"
    )
    
    return transformed_gate

def transform_gate(circuit, instr, qubits, u3_params, block_id, condition=None):
    transformed_gate = create_transformed_gate(instr.operation, u3_params, block_id)
    if condition:
        circuit.append(transformed_gate, qubits).c_if(condition[0], condition[1])
    else:
        circuit.append(transformed_gate, qubits)

def apply_basis_obfuscation(circuit):
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    deferred_measurements = []
    u3_params = generate_random_u3_params()
    
    # Apply basis transformation (directly append U3 gates instead of custom instruction)
    basis_block = get_basis_block(len(circuit.qubits), u3_params, inverse=False, label="Basis")
    for instr in basis_block.data:
        new_circuit.append(instr.operation, instr.qubits)
    
    # Process all instructions
    for instr in circuit.data:
        if instr.operation.name == "measure":
            deferred_measurements.append((instr.operation, instr.qubits, instr.clbits))
        elif instr.operation.name == "barrier":
            new_circuit.append(instr.operation, instr.qubits, instr.clbits)
        elif hasattr(instr.operation, 'condition') and instr.operation.condition:
            # Handle conditional gates (e.g., x.c_if)
            transform_gate(new_circuit, instr, instr.qubits, u3_params, block_id=0, condition=instr.operation.condition)
        else:
            transform_gate(new_circuit, instr, instr.qubits, u3_params, block_id=0)
    

    inv_basis_block = get_basis_block(len(circuit.qubits), u3_params, inverse=True, label="InvBasis")
    for instr in inv_basis_block.data:
        new_circuit.append(instr.operation, instr.qubits)
    
    for meas_op, meas_qubits, meas_clbits in deferred_measurements:
        new_circuit.append(meas_op, meas_qubits, meas_clbits)
    
    return new_circuit

def execute_circuit(circuit):
    sim = AerSimulator()
    execution_times = []
    for _ in range(1):  
        qc = transpile(circuit, sim)
        start_time = time.time()
        job = sim.run(qc, shots=1024)
        end_time = time.time()
        result = job.result()
        execution_times.append(end_time - start_time)
    average_execution_time = sum(execution_times) / len(execution_times)
    return result.get_counts(), average_execution_time

def compare_results(original, obfuscated):
    keys = set(original.keys()).union(obfuscated.keys())
    total = sum(original.values())
    correct = 0
    for key in keys:
        original_count = original.get(key, 0)
        obfuscated_count = obfuscated.get(key, 0)
        correct += min(original_count, obfuscated_count)
    return 100 * correct / total if total > 0 else 0

def compute_tvd_dfc(original_results, obfuscated_results, shots=1024):
    if shots <= 0:
        raise ValueError("Number of shots must be positive")
    
    all_keys = set(original_results).union(obfuscated_results)
    tvd_sum = sum(abs(original_results.get(key, 0) - obfuscated_results.get(key, 0)) for key in all_keys)
    tvd = tvd_sum / (2 * shots)
    
    correct_bitstrings = set(original_results.keys())
    correct_count_sum = sum(obfuscated_results.get(bitstring, 0) for bitstring in correct_bitstrings)
    incorrect_counts = [count for bitstring, count in obfuscated_results.items() 
                       if bitstring not in correct_bitstrings]
    max_incorrect_count = max(incorrect_counts, default=0)
    dfc = (correct_count_sum - max_incorrect_count) / shots if shots > 0 else 0
    
    return tvd, dfc

def basis_obfuscate_and_execute(input_qasm):
    try:
        input_qasm = input_qasm.strip()
        if input_qasm.startswith("OPENQASM 2.0"):
            original_circuit = QuantumCircuit.from_qasm_str(input_qasm)
        elif input_qasm.startswith("OPENQASM 3"):
            original_circuit = qiskit.qasm3.loads(input_qasm)
        else:
            raise ValueError("Invalid QASM version: Must start with 'OPENQASM 2.0;' or 'OPENQASM 3;'")
    except Exception as e:
        print(f"Error parsing QASM: {e}")
        sys.exit(1)

    style = {
        'fontsize': 12,
        'displaycolor': {
            'u3': "#FB0202",  
            'Obf_H': '#FF5733', 'Obf_CNOT': '#FF5733', 'Obf_X': '#FF5733',
            'Obf_Z': '#FF5733', 'Obf_CU': '#FF5733', 'Obf_CP': '#FF5733',
            'Obf_CRY': '#FF5733', 'Obf_SWAP': '#FF5733'
        }
    }

    original_circuit = add_measurements(original_circuit)
    obfuscated_circuit = apply_basis_obfuscation(original_circuit)

    original_results, original_time = execute_circuit(original_circuit)
    obfuscated_results, obfuscated_time = execute_circuit(obfuscated_circuit)
    obfuscated_circuit.draw('mpl', style=style)
    plt.show()

    shots = 1024
    semantic_accuracy = compare_results(original_results, obfuscated_results)
    tvd, dfc = compute_tvd_dfc(original_results, obfuscated_results, shots=shots)

    obfuscated_circuit = transpile(obfuscated_circuit, basis_gates=['u3', 'cx', 'measure'], optimization_level=0)
    
    if input_qasm.startswith("OPENQASM 2.0"):
        obfuscated_qasm_str = qiskit.qasm2.dumps(obfuscated_circuit)
    elif input_qasm.startswith("OPENQASM 3"):
        obfuscated_qasm_str = qiskit.qasm3.dumps(obfuscated_circuit)

    #print(obfuscated_qasm_str)    
    
    return {
        "original_circuit": original_circuit,
        "obfuscated_circuit": obfuscated_circuit,
        "original_results": original_results,
        "obfuscated_results": obfuscated_results,
        "semantic_accuracy": semantic_accuracy,
        "tvd": tvd,
        "dfc": dfc,
        "original_time": original_time,
        "obfuscated_time": obfuscated_time,
    }


if __name__ == "__main__":
    file_path = "QASM Circuits/BV(1011).qasm"  # You may also select any other QASM file from the folder "QASM Circuit"
    with open(file_path, "r") as f:
            test_qasm = f.read() 

    print("\nTesting QASM Circuit:")
    results = basis_obfuscate_and_execute(test_qasm)
    print("\n--- Original Results ---")
    for key, value in results["original_results"].items():
        print(f"Result: {key}, Count: {value}")
    print("\n--- Obfuscated Results ---")
    for key, value in results["obfuscated_results"].items():
        print(f"Result: {key}, Count: {value}")
    print(f"\nSemantic Accuracy: {results['semantic_accuracy']:.2f}%")
    print(f"Original Time: {results['original_time']} s")
    print(f"Obfuscated Time: {results['obfuscated_time']} s")
    print(f"Total Variation Distance (TVD): {results['tvd']}")
    print(f"Degree of Functional Corruption (DFC): {results['dfc']:.4f}")

    all_keys = sorted(set(results["original_results"].keys()).union(results["obfuscated_results"].keys()))
    orig_counts = [results["original_results"].get(k, 0) for k in all_keys]
    obfus_counts = [results["obfuscated_results"].get(k, 0) for k in all_keys]

    x = np.arange(len(all_keys))
    width = 0.35  

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, orig_counts, width, label='Original')
    plt.bar(x + width/2, obfus_counts, width, label='Obfuscated')

    plt.xlabel("Measurement Outcome")
    plt.ylabel("Counts")
    plt.title("Original vs Obfuscated Results")
    plt.xticks(x, all_keys, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()