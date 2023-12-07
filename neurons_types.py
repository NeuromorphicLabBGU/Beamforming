import numpy as np
from sctn.spiking_neuron import create_SCTN, IDENTITY, BINARY


def create_type(
        weights,
        activation_function,
        leakage_factor,
        leakage_period,
        theta,
):
    n = create_SCTN()
    n.synapses_weights = np.array(weights, dtype=np.float64)
    n.activation_function = activation_function
    n.leakage_factor = leakage_factor
    n.leakage_period = leakage_period
    n.theta = theta
    n.membrane_should_reset = False
    return n

def create_type0():
    return create_type(
        weights=[8],
        activation_function=IDENTITY,
        leakage_factor=6,
        leakage_period=127,
        theta=-4,
    )


def create_type1():
    return create_type(
        weights=[127] * 8 + [-128],
        activation_function=IDENTITY,
        leakage_factor=7,
        leakage_period=2,
        theta=-444,
    )


def create_type1a():
    return create_type(
        weights=[255],
        activation_function=BINARY,
        leakage_factor=7,
        leakage_period=2,
        theta=-128,
    )


def create_type2():
    return create_type(
        weights=[255],
        activation_function=BINARY,
        leakage_factor=7,
        leakage_period=2,
        theta=-135,
    )


def create_type3():
    return create_type(
        weights=[127, -127],
        activation_function=BINARY,
        leakage_factor=3,
        leakage_period=0,
        theta=-1,
    )


def create_type4():
    return create_type(
        weights=[1] * 2 + [-1] * 2,
        activation_function=BINARY,
        leakage_factor=6,
        leakage_period=32,
        theta=-1,
    )


def create_type5():
    return create_type(
        weights=[1] * 4 + [-1] * 4,
        activation_function=BINARY,
        leakage_factor=6,
        leakage_period=32,
        theta=-1,
    )


def create_type6():
    return create_type(
        weights=[1] * 8 + [-1] * 8,
        activation_function=BINARY,
        leakage_factor=6,
        leakage_period=32,
        theta=-1,
    )


def create_type7():
    return create_type(
        weights=[1] * 16 + [-1] * 16,
        activation_function=BINARY,
        leakage_factor=6,
        leakage_period=32,
        theta=-1,
    )


def create_type8():
    return create_type(
        weights=[1] * 32 + [-1] * 32,
        activation_function=BINARY,
        leakage_factor=6,
        leakage_period=32,
        theta=-1,
    )


def create_type9():
    w = [15,
         8, 8, 4, 4, 4, 4,
         2, 2, 2, 2, 2, 2,
         2, 2, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1]
    return create_type(
        weights=w,
        activation_function=BINARY,
        leakage_factor=5,
        leakage_period=128,
        theta=-1,
    )
