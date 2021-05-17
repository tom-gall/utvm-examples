import os
import numpy as np
import tvm
import tvm.micro as micro
import logging
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime, utils
from tvm.micro.contrib import zephyr
from tvm import relay

model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
model_file = "./model_ToyCar_quant_fullint_micro_intio.tflite"
model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_file, "rb").read()

######################################################################
# Using the buffer, transform into a tflite model python object
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

version = tflite_model.Version()
print("Model Version: " + str(version))


input_tensor = "input_1_int8"
input_shape = (1,640)
input_dtype = "int8"

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

logging.basicConfig(level=logging.DEBUG)

# stm32f746xx || host
#TARGET = tvm.target.target.micro("host")
TARGET = tvm.target.target.micro("stm32f746xx")

with tvm.transform.PassContext(
    opt_level=3, config={"tir.disable_vectorize": True},
):
    graph, c_mod, c_params = relay.build(mod, target=TARGET, params=params)

workspace = tvm.micro.Workspace(debug=True)

# 'project_dir' set from hardcoded path
# github : https://github.com/tom-gall/zephyr-runtime.git
# project_dir = os.path.join("/home/tgall/tvm/", "zephyr-runtime")

# 'project_dir' based upon relative Git repo
# git_cmd = ["git", "rev-parse", "--show-toplevel"]
# repo_root = subprocess.check_output(git_cmd, encoding="utf-8").strip()
# project_dir = f"{repo_root}/apps/microtvm/zephyr/demo_runtime"

# 'project_dir' based upon TVM_HOME env variable
repo_root = os.environ.get("TVM_HOME")
if not repo_root:
	print(f'ERROR: Please set env variable TVM_HOME correctly')
	exit()
project_dir = f"{repo_root}/apps/microtvm/zephyr/demo_runtime"

compiler = zephyr.ZephyrCompiler(project_dir=project_dir,
        board="stm32f746g_disco",
        zephyr_toolchain_variant="zephyr",
        )
opts = tvm.micro.default_options(f"{project_dir}/crt")
#opts["bin_opts"]["ccflags"] = ["-std=gnu++14"]
#opts["lib_opts"]["ccflags"] = ["-std=gnu++14"]

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
    #extra_libs=[os.path.join(tvm.micro.build.CRT_ROOT_DIR, "memory")],
)

flasher = compiler.flasher()
with tvm.micro.Session(binary=micro_binary, flasher=flasher) as session:
    graph_mod = tvm.micro.create_local_graph_runtime(
        graph, session.get_system_lib(), session.context
    )

    # Set the model parameters using the lowered parameters produced by `relay.build`.
    graph_mod.set_input(**c_params)

    # [1,640] - input
    graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="int8")))
    graph_mod.run()

    tvm_output = graph_mod.get_output(0).asnumpy()
    print("result is: " + str(tvm_output))

