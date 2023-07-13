from typing import Optional, List

import os
import time
import argparse
import numpy as np

import torch
from transformers import AutoModelForCausalLM

import tvm
from mlc_llm import utils

# from mlc_llm.conversation import SeparatorStyle, conv_templates, compute_skip_echo_len
from utils import get_tokenizer, get_pytorch_model, get_tvm_model, sample_top_p


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# NOTE: "all" is currently not supported due to GPU OOM issue in creating multiple model wrappers.
BENCHMARK_MODES = ["tvm", "torch-eager", "torch-inductor", "llama-cpp"]


def instrument_nvtx_range(func, name, before_run, *args):
    if before_run:
        torch.cuda.nvtx.range_push(f"{name}")
    else:
        torch.cuda.nvtx.range_pop()


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="auto",
        help='The name of the model to build. If it is "auto", we will automatically set the '
        'model name according to "--model-path", "hf-path" or the model folders under '
        '"--artifact-path/models"',
    )
    args.add_argument(
        "--quantization",
        type=str,
        choices=[*utils.quantization_schemes.keys()],
        default=list(utils.quantization_schemes.keys())[0],
    )
    args.add_argument(
        "--prompt",
        type=str,
        default="Repeat the following paragraph exactly: Carlos Alcaraz is a professional tennis player from Spain. He was born on April 12, 1997, and has been playing tennis since he was a child. Alcaraz is known for his aggressive playing style and his ability to come back from difficult situations. He has won several junior tournaments and has also played on the ATP Tour. Alcaraz He has a career high ranking in men’s singles by the Association of Tennis Professionals of world No. 1 achieved on 12 September 2022. He is currently ranked world No. 2 by the ATP. He is known for his strong serve and powerful groundstrokes. He is also known for his positive attitude and his determination to succeed on the court.",
    )
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--ggml-file-name", type=str, default="ggml-model-f16.bin")
    args.add_argument("--max-gen-len", type=int, default=2048)
    args.add_argument(
        "--benchmark-mode",
        type=str,
        choices=BENCHMARK_MODES,
        default=BENCHMARK_MODES[0],
    )
    args.add_argument("--skip_sampling", action="store_true", default=False)
    args.add_argument("--num-warm-up", type=int, default=5)
    args.add_argument("--num-measurements", type=int, default=10)
    args.add_argument("--num-input-tokens", type=int, default=32)
    args.add_argument("--num-output-tokens", type=int, default=32)

    parsed = args.parse_args()
    utils.argparse_postproc_common(parsed)

    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}"
    )
    parsed.ggml_file_path = os.path.join(parsed.model_path, parsed.ggml_file_name)
    return parsed


class ModelWrapper:
    def __init__(self, tokenizer, max_gen_len: int, conv_template):
        self.name = None
        self.tokenizer = tokenizer
        self.max_gen_len = max_gen_len
        self.conv_template = conv_template

    # This function forces device sync
    def sync(self):
        assert 0, "Not intended to call directly"

    # This prepares model for benchmarking.
    # One of its important jobs is to clear kv cache.
    def prep_model(self):
        assert 0, "Not intended to call directly"

    # This function implements the core of the pipeline for benchmarking.
    # This excludes the interaction with tokenizer (e.g., text->token, token->text)
    # Currently, sampling is implemented as greedy and you can also skip this by
    # using `skip_sampling`.
    def benchmark_core(self, num_input_tokens, num_output_tokens, skip_sampling=False):
        assert 0, "Not intended to call directly"

    # This function is for e2e pipeline. Currently, mainly used for the sanity check
    # of the output texts.
    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
    ):
        raise NotImplementedError


class TvmModelWrapper(ModelWrapper):
    def __init__(
        self,
        tokenizer,
        max_gen_len,
        conv_template,
        artifact_path,
        model_name,
        quant,
        dtype,
        tvm_device,
        torch_device=("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(tokenizer, max_gen_len, conv_template)

        self.name = "tvm_model_wrapper"
        self.artifact_path = artifact_path
        self.model_name = model_name
        self.dtype = dtype
        tvm_ex = tvm.runtime.load_module(
            f"{artifact_path}/{model_name}-{quant}-{tvm_device}.so"
        )
        self.tvm_device = tvm.device(tvm_device)
        self.torch_device = torch.device(torch_device)

        self.const_params = utils.load_params(artifact_path, self.tvm_device)
        self.vm = tvm.relax.VirtualMachine(tvm_ex, self.tvm_device)
        self.prep_model()

    def sync(self):
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize()

        if str(self.tvm_device) != "cpu":
            self.tvm_device.sync()

    def prep_model(self, *args, **kwarg):
        self.model = get_tvm_model(self.const_params, self.vm)

    def benchmark_core(self, num_input_tokens, num_output_tokens, skip_sampling=False):
        total_len = num_input_tokens + num_output_tokens
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(self.torch_device)
        )

        start_pos = num_input_tokens
        next_token = None

        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                tok = tvm.nd.from_dlpack(tokens[:, :cur_pos])
                logits = self.model(tok)
            else:
                tok = tvm.nd.from_dlpack(next_token)
                logits = self.model(tok)

            # NOTE:
            # There are three methods to work with torch ops.
            # Method 1: tvm GPU -> torch GPU
            #    logits = torch.from_dlpack(logits)
            # Method 2: tvm GPU-> tvm CPU -> torch CPU
            #    logits = tvm.nd.array(logits.numpy(), tvm.cpu())
            #    logits = torch.from_dlpack(logits)
            # Method 3: tvm GPU -> numpy CPU -> torch CPU
            #    logits = torch.from_numpy(logits.numpy())
            #
            # For logits of [1,1,32k] = 128KB (our case), conversion on GPU (Method1) takes ~1sec,
            # which could be non-neglibile for short response.
            # However, conversion on CPU (Method 2&3) is very cheap even with the data copy from GPU to CPU.
            # Rather than addressing this issue, we choose Method3 by default for now.

            if skip_sampling:
                continue

            if str(self.torch_device) == "cpu":
                logits = torch.from_numpy(logits.numpy())
            else:
                logits = torch.from_dlpack(logits)

            logits = logits[:, -1, :]
            # Use greedy
            next_token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)

    def generate(
        self,
        prompt_tokens,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
    ):
        prompt_len = len(prompt_tokens)
        total_len = self.max_gen_len + prompt_len
        # TODO: Replace Torch ops to TVM
        # Issue 1: `tokens` requires in-place update.
        # Issue 2: p-sampling use random generator which might have side effect.
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(self.torch_device)
        )

        tokens[0, :prompt_len] = (
            torch.tensor(prompt_tokens).to(torch.int32).to(self.torch_device)
        )

        start_pos = prompt_len
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                # TODO: switch to the below when Eric's PR is merged.
                tok = tvm.nd.from_dlpack(tokens[:, :cur_pos])
                # tok = tvm.nd.array(tokens[:, :cur_pos].numpy(), self.tvm_device)
                logits = self.model(tok)
            else:
                # TODO: switch to the below when Eric's PR is merged.
                tok = tvm.nd.from_dlpack(tokens[:, cur_pos - 1 : cur_pos])
                # tok = tvm.nd.array(
                #    tokens[:, cur_pos - 1 : cur_pos].numpy(), self.tvm_device
                # )
                logits = self.model(tok)

            # NOTE:
            # There are three methods to work with torch ops.
            # Method 1: tvm GPU -> torch GPU
            #    logits = torch.from_dlpack(logits)
            # Method 2: tvm GPU-> tvm CPU -> torch CPU
            #    logits = tvm.nd.array(logits.numpy(), tvm.cpu())
            #    logits = torch.from_dlpack(logits)
            # Method 3: tvm GPU -> numpy CPU -> torch CPU
            #    logits = torch.from_numpy(logits.numpy())
            #
            # For logits of [1,1,32k] = 128KB (our case), conversion on GPU (Method1) takes ~1sec,
            # which could be non-neglibile for short response.
            # However, conversion on CPU (Method 2&3) is very cheap even with the data copy from GPU to CPU.
            # Rather than addressing this issue, we choose Method3 by default for now.

            if str(self.torch_device) == "cpu":
                logits = torch.from_numpy(logits.numpy())
            else:
                logits = torch.from_dlpack(logits)

            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax((logits / temperature).to(torch.float32), dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            tokens[:, cur_pos] = next_token

            stopped = next_token[0] == self.tokenizer.eos_token_id

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == self.max_gen_len - 1 or stopped:
                output = tokens[0, : cur_pos + 1]
                output = self.tokenizer.decode(output, skip_special_tokens=True)
                pos = output.rfind(stop_str, prompt_len)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output

            if stopped:
                break


class TorchModelWrapper(ModelWrapper):
    def __init__(
        self,
        tokenizer,
        max_gen_len,
        conv_template,
        model_path,
        dtype,
        torch_mode,
        torch_device=("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(tokenizer, max_gen_len, conv_template)
        self.name = f"torch_{torch_mode}_model_wrapper"
        self.torch_device = torch.device(torch_device)
        self.model_path = model_path
        self.dtype = dtype
        self.torch_mode = torch_mode

        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        if dtype == "float16":
            model = model.to(torch.float16)
        self._model = model.to(torch_device)
        self.prep_model()

    def prep_model(self, *args, **kwarg):
        model = get_pytorch_model(self._model)
        self.model = torch.compile(model, backend=self.torch_mode, dynamic=True)

    def sync(self):
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize()

    def benchmark_core(self, num_input_tokens, num_output_tokens, skip_sampling=False):
        total_len = num_input_tokens + num_output_tokens
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(self.torch_device)
        )
        past_key_values = None
        for cur_pos in range(num_input_tokens, total_len):
            if cur_pos == num_input_tokens:
                logits, past_key_values = self.model(
                    inputs=tokens[:, :cur_pos], past_key_values=past_key_values
                )
            else:
                logits, past_key_values = self.model(
                    inputs=tokens[:, cur_pos - 1 : cur_pos],
                    past_key_values=past_key_values,
                )

            if skip_sampling:
                continue

            logits = logits[:, -1, :]
            # Use greedy
            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token

    def generate(
        self,
        prompt_tokens,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
    ):
        prompt_len = len(prompt_tokens)
        total_len = self.max_gen_len + prompt_len
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(self.torch_device)
        )
        tokens[0, :prompt_len] = torch.tensor(prompt_tokens).to(torch.int32)
        start_pos = prompt_len
        past_key_values = None
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                logits, past_key_values = self.model(
                    inputs=tokens[:, :cur_pos], past_key_values=past_key_values
                )
            else:
                logits, past_key_values = self.model(
                    inputs=tokens[:, cur_pos - 1 : cur_pos],
                    past_key_values=past_key_values,
                )
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax((logits / temperature).to(torch.float32), dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token

            stopped = next_token[0] == self.tokenizer.eos_token_id

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == self.max_gen_len - 1 or stopped:
                output = self.tokenizer.decode(
                    tokens[0, : cur_pos + 1], skip_special_tokens=True
                )
                pos = output.rfind(stop_str, prompt_len)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output
            if stopped:
                break


class LlamaCppModelWrapper(ModelWrapper):
    name: str = "llama_cpp_model_wrapper"  # name of the model wrapper
    ggml_file_path: str  # path to the ggml model binary path
    model: "Llama"  # llama model object
    tokens: List["llama_token"]  # llama tokens

    def __init__(
        self,
        tokenizer,
        max_gen_len,
        conv_template,
        ggml_file_path: str,  # path to the ggml model binary path
        n_ctx: int = 2048,  # n_ctx text context
        n_parts: int = -1,  # -1 for default
        n_gpu_layers: int = 32,  # number of layers to store in VRAM
        seed: int = 1337,  # RNG seed, 0 for random
        f16_kv: bool = True,  # use fp16 for KV cache
        logits_all: bool = False,  # the llama_eval() call computes all logits, not just the last one
        vocab_only: bool = False,  # only load the vocabulary, no weights
        use_mmap: bool = True,  # use mmap if possible
        use_mlock: bool = False,  # force system to keep model in RAM
        embedding: bool = False,  # embedding mode only
        n_threads: Optional[int] = None,
        n_batch: int = 512,
        last_n_tokens_size: int = 64,
        lora_base: Optional[str] = None,
        lora_path: Optional[str] = None,
        verbose: bool = True,
    ):
        from llama_cpp import Llama, llama_token

        super().__init__(tokenizer, max_gen_len, conv_template)

        self.name = f"llama_cpp_model_wrapper"
        self.ggml_file_path = ggml_file_path
        self.model = Llama(
            ggml_file_path,
            n_ctx=n_ctx,
            n_parts=n_parts,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            f16_kv=f16_kv,
            logits_all=logits_all,
            vocab_only=vocab_only,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            embedding=embedding,
            n_threads=n_threads,
            n_batch=n_batch,
            last_n_tokens_size=last_n_tokens_size,
            lora_base=lora_base,
            lora_path=lora_path,
            verbose=verbose,
        )

    def sync(self):
        pass

    def prep_model(self, num_input_tokens: int, num_output_tokens: int, *args, **kwarg):
        self.model.reset()
        self.tokens = self.model.tokenize(
            (" ".join(["test" for _ in range(num_input_tokens)])).encode("utf-8")
        )

    def benchmark_core(self, num_input_tokens: int, num_output_tokens: int):
        tokens = self.model.generate(
            self.tokens, top_k=40, top_p=0.95, temp=0.8, repeat_penalty=1.1
        )
        for _ in range(num_output_tokens):
            next(tokens)

    def generate(
        self,
        prompt: str = "Q: Name the planets in the solar system? A: ",
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
    ):
        output = self.model(
            prompt,
            temperature=temperature,
            top_p=top_p,
            stop=stop_str,
            stream_interval=stream_interval,
        )
        return iter(output)


# Benchmark single-round conv
# def benchmark_e2e_chat(model_wrapper, prompt, enable_print=False):
#     conv = conv_templates[model_wrapper.conv_template].copy()
#     keep_first_token = True

#     conv.append_message(conv.roles[0], prompt)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     model_wrapper.sync()
#     t0 = time.time()
#     prompt_tokens = model_wrapper.tokenizer.encode(prompt)
#     if not keep_first_token:
#         prompt_tokens = prompt_tokens[1:]

#     pre = 0
#     skip_echo_len = compute_skip_echo_len(model_wrapper.conv_template, conv, prompt)
#     outputs = None
#     for outputs in model_wrapper.generate(
#         prompt_tokens,
#         temperature=0,  # Use greedy to make it deterministic for benchmarking
#         stop_str=conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
#         # stream_interval=2048,  # To output at once
#     ):
#         outputs = outputs[skip_echo_len:].strip()
#         outputs = outputs.split(" ")
#         now = len(outputs)
#         if now - 1 > pre:
#             if enable_print:
#                 print(
#                     Colors.OKBLUE + " ".join(outputs[pre : now - 1]) + Colors.ENDC,
#                     end=" ",
#                     flush=True,
#                 )
#             pre = now - 1
#     if enable_print:
#         print(
#             Colors.OKBLUE + " ".join(outputs[pre:]) + Colors.ENDC,
#             flush=True,
#         )

#     model_wrapper.sync()
#     t1 = time.time()
#     assert now == len(outputs)
#     elapsed = t1 - t0
#     num_input_prompt_tokens, num_decoded_output_tokens = len(prompt_tokens), len(
#         outputs
#     )

#     return num_input_prompt_tokens, num_decoded_output_tokens, elapsed


def benchmark_core_chat(
    model_wrapper, num_input_tokens, num_output_tokens, skip_sampling
):
    # Clear kv cache and prepare the model
    model_wrapper.prep_model(num_input_tokens, num_output_tokens)
    model_wrapper.sync()
    t0 = time.time()
    model_wrapper.benchmark_core(num_input_tokens, num_output_tokens, skip_sampling)
    model_wrapper.sync()
    t1 = time.time()
    return t1 - t0


def benchmark(
    model_wrapper,
    num_input_tokens,
    num_output_tokens,
    num_warm_up,
    num_measurement,
    skip_sampling=False,
    percentiles=[50, 90, 99],
):
    # Warm-up
    for _ in range(num_warm_up):
        benchmark_core_chat(
            model_wrapper, num_input_tokens, num_output_tokens, skip_sampling
        )

    # Actual measurement
    elapsed = []
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_measurement):
        torch.cuda.nvtx.range_push(f"iteration")
        elapsed.append(
            benchmark_core_chat(
                model_wrapper, num_input_tokens, num_output_tokens, skip_sampling
            )
        )
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    elapsed = [np.percentile(elapsed, p) for p in percentiles]
    tok_per_sec = [num_output_tokens / e for e in elapsed]
    return elapsed, tok_per_sec


def get_model_wrapper(mode, tokenizer, ARGS):
    if mode == "tvm":
        return TvmModelWrapper(
            tokenizer,
            ARGS.max_gen_len,
            ARGS.conv_template,
            ARGS.artifact_path,
            ARGS.model,
            ARGS.quantization.name,
            ARGS.quantization.model_dtype,
            tvm_device=ARGS.device_name,
            torch_device="cuda",  # TODO: change to "cuda" when dlpack conversion works.
        )
    elif mode.startswith("torch-"):
        return TorchModelWrapper(
            tokenizer,
            ARGS.max_gen_len,
            ARGS.conv_template,
            ARGS.model_path,
            ARGS.quantization.model_dtype,
            torch_device="cuda",
            torch_mode=mode[6:],
        )
    elif mode == "llama-cpp":
        return LlamaCppModelWrapper(
            tokenizer, ARGS.max_gen_len, ARGS.conv_template, ARGS.ggml_file_path
        )
    raise ValueError(f"Unknown mode {mode}")


if __name__ == "__main__":
    ARGS = _parse_args()
    # Torch setup
    torch.set_float32_matmul_precision("high")

    # Tokenizer setup
    tokenizer = get_tokenizer(ARGS.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_wrapper = get_model_wrapper(ARGS.benchmark_mode, tokenizer, ARGS)
    percentiles = [50, 90, 99]

    # Example of e2e chat. Disable if you want to try
    # benchmark_e2e_chat(model_wrapper, ARGS.prompt, True)

    # TODO: Extend to the list of repr input pairs
    pairs = [(ARGS.num_input_tokens, ARGS.num_output_tokens)]
    for num_input_tokens, num_output_tokens in pairs:
        elapsed, tok_per_sec = benchmark(
            model_wrapper,
            num_input_tokens,
            num_output_tokens,
            ARGS.num_warm_up,
            ARGS.num_measurements,
            percentiles=percentiles,
            skip_sampling=ARGS.skip_sampling,
        )

        print("|{:^15}|{:^12}|{:^12}|".format("mode", "seqlen", "genlen"), end="")
        for p in percentiles:
            print("{:^12}|{:^12}|".format(f"p{p}: sec", f"p{p}: tok/s"), end="")
        print("")
        print(
            "|{:^15}|{:^12}|{:^12}|".format(
                ARGS.benchmark_mode,
                num_input_tokens,
                num_output_tokens,
            ),
            end="",
        )
        for i in range(len(percentiles)):
            print("{:^12.3f}|{:^12.3f}|".format(elapsed[i], tok_per_sec[i]), end="")
        print("")
    del model_wrapper
