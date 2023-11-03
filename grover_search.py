import numpy as np
from qiskit import QuantumCircuit, Aer, execute

def grover_search(n, marked):
    # Create a quantum circuit on n qubits
    circuit = QuantumCircuit(n, n)

    # Apply Hadamard gates to all qubits
    circuit.h(range(n))

    # Apply the Oracle
    marked_oracle(circuit, marked)

    # Apply the Diffusion Operator
    diffusion_operator(circuit)

    # Measure all qubits
    circuit.measure(range(n), range(n))

    return circuit

def marked_oracle(circuit, marked):
    # Apply X gate to the marked state
    for state in marked:
        circuit.x(state)

    # Apply multi-controlled Z gate
    circuit.h(marked[-1])
    circuit.mct(marked[:-1], marked[-1])
    circuit.h(marked[-1])

    # Apply X gate to the marked state again
    for state in marked:
        circuit.x(state)

def diffusion_operator(circuit):
    # Apply Hadamard gates to all qubits
    circuit.h(range(n))

    # Apply X gates to all qubits
    circuit.x(range(n))

    # Apply multi-controlled Z gate
    circuit.h(n-1)
    circuit.mct(list(range(n-1)), n-1)
    circuit.h(n-1)

    # Apply X gates to all qubits
    circuit.x(range(n))

    # Apply Hadamard gates to all qubits
    circuit.h(range(n))

# Define the number of qubits and the marked states
n = 5
marked = [0, 1, 2, 3]

# Create the Grover's circuit
grover_circuit = grover_search(n, marked)

# Simulate the circuit using the Qiskit
