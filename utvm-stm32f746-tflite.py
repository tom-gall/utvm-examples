import os
import numpy as np
import tvm
import logging
import tvm.micro as micro
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime, utils
from tvm.micro.contrib import zephyr
from tvm import relay

model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
model_file = "sine_model.tflite"
model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_path, "rb").read()

######################################################################
# Using the buffer, transform into a tflite model python object
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

logging.basicConfig(level=logging.DEBUG)

# stm32f746xx || host
#TARGET = tvm.target.target.micro("host")
TARGET = tvm.target.target.micro("stm32f746xx")

with tvm.transform.PassContext(
    opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["FuseOps"]
):
    graph, c_mod, c_params = relay.build(mod, target=TARGET, params=params)

workspace = tvm.micro.Workspace(debug=True)


# github : https://github.com/tom-gall/zephyr-runtime.git
project_dir = os.path.join("/home/tgall/tvm/", "zephyr-runtime")

compiler = zephyr.ZephyrCompiler(project_dir=project_dir,
        board="stm32f746g_disco",
        zephyr_toolchain_variant="zephyr",
        )
opts = tvm.micro.default_options(f"{project_dir}/crt")
opts["bin_opts"]["ccflags"] = ["-std=gnu++14"]
opts["lib_opts"]["ccflags"] = ["-std=gnu++14"]



micro_binary = tvm.micro.build_static_runtime(
    # the x86 compiler *expects* you to give the exact same dictionary for both
    # lib_opts and bin_opts. so the library compiler is mutating lib_opts and
    # the binary compiler is expecting those mutations to be in bin_opts.
    # TODO(weberlo) fix this very bizarre behavior
    workspace,
    compiler,
    c_mod,
    lib_opts=opts["bin_opts"],
    bin_opts=opts["bin_opts"],
)

flasher = compiler.flasher()
with tvm.micro.Session(binary=micro_binary, flasher=flasher) as session:
    graph_mod = tvm.micro.create_local_graph_runtime(
        graph, session.get_system_lib(), session.context
    )

    # Set the model parameters using the lowered parameters produced by `relay.build`.
    graph_mod.set_input(**c_params)

    # The model consumes a single float32 value and returns a predicted sine value.  To pass the
    # input value we construct a tvm.nd.array object with a single contrived number as input. For
    # this model values of 0 to 2Pi are acceptable.
    graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))
    graph_mod.run()

    tvm_output = graph_mod.get_output(0).asnumpy()
    print("result is: " + str(tvm_output))

