"""
Packages used to obtain flops from GNN and DistillNet
"""

###GNN
from typing import Optional, Union
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
import keras_flops.flops_registory

def get_flops(model: Union[Model, Sequential], batch_size: Optional[int] = None) -> int:
    """
    Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v1 api.
    """
    if not isinstance(model, (Sequential, Model)):
        raise KeyError(
            "model arguments must be tf.keras.Model or tf.keras.Sequential instanse"
        )

    if batch_size is None:
        batch_size = 1

    # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
    # FLOPS depends on batch size
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPS with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    # print(frozen_func.graph.get_operations())
    return flops.total_float_ops

###DistillNet
from deepspeed.profiling.flops_profiler import get_model_profile
#takes pytorch model state dict as input
flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=(9000, 16), # input shape to the model. In our case 1 event i.e. 9k particles times 16 input features
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling

