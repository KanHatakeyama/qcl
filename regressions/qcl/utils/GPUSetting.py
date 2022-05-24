from qulacs import QuantumState
"""
If you want to use GPU, turn on GPU_FLAG
NOTE: use of gpu may be slower for small-size calculations
"""
GPU_FLAG = False

if GPU_FLAG:
    try:
        from qulacs import QuantumStateGpu
        QuantumState = QuantumStateGpu
        print("use gpu")
    except:
        pass
