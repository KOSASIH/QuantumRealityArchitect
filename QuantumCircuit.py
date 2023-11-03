import numpy as np

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state_vector[0] = 1.0

    def apply_gate(self, gate, target_qubits):
        gate_matrix = gate.get_matrix()

        if len(target_qubits) == 1:
            target_qubit = target_qubits[0]
            gate_matrix = np.kron(np.eye(2**(self.num_qubits - target_qubit - 1)), gate_matrix)
            gate_matrix = np.kron(gate_matrix, np.eye(2**target_qubit))
        else:
            target_qubits = sorted(target_qubits, reverse=True)
            gate_matrix = np.kron(np.eye(2**self.num_qubits), gate_matrix)

            for target_qubit in target_qubits:
                gate_matrix = np.delete(gate_matrix, range(2**target_qubit, 2**(target_qubit + 1)), axis=0)
                gate_matrix = np.delete(gate_matrix, range(2**target_qubit, 2**(target_qubit + 1)), axis=1)

        self.state_vector = np.dot(gate_matrix, self.state_vector)

    def measure(self, num_shots=1):
        probabilities = np.abs(self.state_vector)**2
        measurement_outcomes = np.random.choice(range(len(self.state_vector)), size=num_shots, p=probabilities)
        return measurement_outcomes

    def get_state_vector(self):
        return self.state_vector

class Gate:
    def __init__(self, matrix):
        self.matrix = matrix

    def get_matrix(self):
        return self.matrix
