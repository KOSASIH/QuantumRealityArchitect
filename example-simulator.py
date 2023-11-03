# Create a quantum circuit with 2 qubits
qc = QuantumCircuit(2)

# Apply a Hadamard gate to the first qubit
h_gate = Gate(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
qc.apply_gate(h_gate, [0])

# Apply a CNOT gate to the second qubit controlled by the first qubit
cnot_gate = Gate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
qc.apply_gate(cnot_gate, [0, 1])

# Measure the qubits
measurement_outcomes = qc.measure()

print("Measurement outcomes:", measurement_outcomes)
print("State vector:", qc.get_state_vector())
