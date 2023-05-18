# pylint: disable=missing-docstring,invalid-name
import argparse
from typing import List, Tuple

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax

import re
import sys
import shutil
import subprocess

def get_llvm_target() -> tvm.target.Target:
    """Extract fully featured llvm target for current device.

    Returns
    -------
    target : tvm.target.Target
        A TVM target that fully describes the current devices CPU.
    """
    # If we cant find llc, we wont be able to extract more information.
    if shutil.which("llc") is None:
        print(
            "Could not find llc, falling back to default llvm. "
            "Consider installing llc for better performance"
        )
        return "llvm"

    # Get host information from llc
    cpu_info = subprocess.check_output("llc --version", shell=True).decode()

    # Parse out cpu line
    cpu = re.search("(?<=Host CPU: ).+", cpu_info).group(0)

    # Next extract attribute string.
    platform = sys.platform
    # Linux
    if platform not in ["linux", "linux2"]:
        raise ValueError("Platform %s is not supported." % platform)
    output = subprocess.check_output("lscpu", shell=True).decode()
    # The output of lscpu produces a bunch of lines with the format
    # "Title: Value". This pattern matches both the title and value
    # parts of each line so that we can construct a dictionary.
    pattern = r"^([^:]+):\s+(.*)$"
    cpu_info = {}

    for line in output.splitlines():
        match = re.match(pattern, line)
        if match:
            key = match.group(1)
            value = match.group(2)
            cpu_info[key] = value.lower().strip()

    features = cpu_info["Flags"].split(" ")
    march = cpu_info["Architecture"]
    cores = cpu_info["Core(s) per socket"]
    sockets = cpu_info["Socket(s)"]
    total_cores = str(int(cores) * int(sockets))
    # Special case for x86_64 mismatch between underscore and hyphen
    if march == "x86_64":
        march = "x86-64"

    # Now we'll extract the architecture of the target.
    output = subprocess.check_output("llc --version", shell=True).decode()
    # Remove header.
    march_options = re.search("(?<=Registered Targets:).*", output, re.DOTALL).group(0)
    march_list = [m.strip().split(" ")[0] for m in march_options.split("\n") if m]
    valid_march = march in march_list
    # Build the base target.
    host_target = (
        subprocess.check_output("llvm-config --host-target", shell=True).decode().strip("\n")
    )
    target = f"llvm -mcpu={cpu} -mtriple={host_target} -num-cores={total_cores}"

    # If possible, add more attribute information.
    if not valid_march:
        return tvm.target.Target(target)

    # Get list of valid attributes for the target architecture.
    attrs_info = subprocess.check_output(
        "llc -march=%s -mattr=help" % march, shell=True, stderr=subprocess.STDOUT
    ).decode()
    supported_attrs = re.search(
        r"(?<=Available features for this target:).*(?=Use \+feature to enable a feature)",
        attrs_info,
        re.DOTALL,
    ).group(0)
    # Find which features are supported attrs.
    attrs_list = [attr.strip().split(" ")[0] for attr in supported_attrs.split("\n")]
    attrs = [f for f in features if f in attrs_list]

    # Compuse attributes into valid string.
    attrs_string = ",".join(f"+{a}" for a in attrs)

    # Now we can add more information to the llvm target.
    target = "%s -mattr=%s" % (target, attrs_string)

    return tvm.target.Target(target)

def argparse_add_common(args: argparse.ArgumentParser) -> None:
    args.add_argument(
        "--model",
        type=str,
        default="vicuna-v1-7b",
        choices=[
            "vicuna-v1-7b",
            "dolly-v2-3b",
            "dolly-v2-7b",
            "dolly-v2-12b",
            "stablelm-tuned-alpha-3b",
            "stablelm-tuned-alpha-7b",
            "llama-30b",
        ],
    )
    args.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
    )


def argparse_postproc_common(args: argparse.Namespace) -> None:
    if hasattr(args, "device_name"):
        if args.device_name == "auto":
            if tvm.cuda().exist:
                args.device_name = "cuda"
            elif tvm.metal().exist:
                args.device_name = "metal"
            else:
                raise ValueError("Cannot auto deduce device-name, please set it")
    if args.model.startswith("vicuna-") or args.model.startswith("llama-"):
        from mlc_llm.relax_model import llama  # pylint: disable=import-outside-toplevel

        args.conv_template = "vicuna_v1.1"

    elif args.model.startswith("dolly-") or args.model.startswith("stablelm-"):
        from mlc_llm.relax_model import (
            gpt_neox,
        )  # pylint: disable=import-outside-toplevel

        if args.model.startswith("dolly-"):
            args.conv_template = "dolly"
        elif args.model.startswith("stablelm-"):
            args.conv_template = "stablelm"
    else:
        raise ValueError(f"Model {args.model} not supportqed")


def split_transform_deploy_mod(
    mod: tvm.IRModule, model_names: List[str]
) -> Tuple[tvm.IRModule, tvm.IRModule]:
    mod_transform = tvm.IRModule()
    mod_deploy = tvm.IRModule()

    transform_func_name = None
    gv_names = [gv.name_hint for gv in mod.get_global_vars()]
    for name in model_names:
        if name + "_transform_params" in gv_names:
            transform_func_name = name + "_transform_params"
    assert transform_func_name is not None

    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
            mod_deploy[gv] = func
        elif gv.name_hint == transform_func_name:
            mod_transform[gv] = func
        else:
            mod_deploy[gv] = func

    mod_transform = relax.transform.DeadCodeElimination([transform_func_name])(
        mod_transform
    )
    mod_deploy = relax.transform.DeadCodeElimination(model_names)(mod_deploy)

    # Copy the runtime module from external codegen
    mod_deploy = mod_deploy.with_attrs(
        {
            "external_mods": mod.get_attr("external_mods"),
            "const_name_to_constant": mod.get_attr("const_name_to_constant"),
        }
    )

    return mod_transform, mod_deploy


def transform_params(
    mod_transform: tvm.IRModule, model_params: List[tvm.nd.NDArray]
) -> List[tvm.nd.NDArray]:
    import time
    transform_func_name = None
    for gv, func in mod_transform.functions.items():
        if isinstance(func, relax.Function):
            transform_func_name = gv.name_hint
    assert transform_func_name is not None

    ex = relax.build(mod_transform, target=get_llvm_target())
    vm = relax.vm.VirtualMachine(ex, tvm.cpu())

    t0 = time.time()
    res = vm[transform_func_name](model_params)
    t1 = time.time()
    print("Time spent transforming params: ", t1 - t0, " seconds")
    return res


def save_params(params: List[tvm.nd.NDArray], artifact_path: str) -> None:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    meta_data = {}
    param_dict = {}
    meta_data["ParamSize"] = len(params)
    for i, nd in enumerate(params):
        param_dict[f"param_{i}"] = nd
    tvmjs.dump_ndarray_cache(
        param_dict, f"{artifact_path}/params", meta_data=meta_data, encode_format="raw"
    )


def load_params(artifact_path: str, device) -> List[tvm.nd.NDArray]:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


def build_model_from_log(relax_mod, target, log_dir):
    db = ms.database.create(work_dir=log_dir)
    with target, db, tvm.transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)
    return relax_mod
