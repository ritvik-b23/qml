import pennylane as qml

dev = qml.device("lightning.gpu", wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0))

print(circuit())
