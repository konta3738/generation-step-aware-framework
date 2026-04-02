#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
import typing as t
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoModelForPreTraining, AutoConfig, LlamaForCausalLM, XGLMForCausalLM, BloomForCausalLM, SeamlessM4Tv2Model

from typing import Dict, Optional, Set, Tuple, Union

# --- NEW: memory/dtype helpers ---
import os
from packaging import version

def _gpu_capability_allows_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8  # Ampere+ (A100, etc.)

def _select_torch_dtype():
    # Env override: SC_TORCH_DTYPE in {"bf16","fp16","float32"}
    override = os.environ.get("SC_TORCH_DTYPE", "").lower()
    if override in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if override in {"fp16", "float16", "half"}:
        return torch.float16
    if override in {"fp32", "float32"}:
        return torch.float32

    # Default heuristic
    if _gpu_capability_allows_bf16():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

def _want_8bit() -> bool:
    # Set SC_LOAD_IN_8BIT=1 to force 8bit if bitsandbytes is present
    return os.environ.get("SC_LOAD_IN_8BIT", "0") in {"1", "true", "True"}

def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except Exception:
        return False

MODEL_INPUT_FIELDS = ["input_ids", "attention_mask"]
LABELS_FIELD = "labels"


@dataclass(frozen=True)
class ResponseInfo:
    """
    Information about of a model's response.

    A response is the output tensor of a model operation (ie. layer or a module depending on your
    deep-learning framework). Note that an operation may have more than one response.
    """

    name: str
    """Name of the response."""

    dtype: np.dtype
    """Data type of the response."""

    shape: t.Tuple[t.Optional[int], ...]
    """Shape of the response. The first dimension will generally be `None`."""

    layer: "ResponseInfo.Layer"
    """Details about the layer that generates this response."""

    @dataclass(frozen=True)
    class Layer:
        """
        Class to hold information about the layer that generated the response.
        """

        name: str
        """Name of the layer that generated the response."""

        kind: str
        """Type of layer."""


class TorchModel:
    """
    Class wrapping a Pytorch model so that we can read intermediate responses from it.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        input_size: t.Mapping[str, t.Tuple],
        input_type: t.Mapping[str, torch.dtype],
        name: str,
        device: str = None,

    ) -> None:
        """
        Wraps a pytorch module to enable reading intermediate responses.
        Args:
            module: A pytorch nn.module holding the model to be wrapped.
            input_size: A dict with model input names as keys and the expected sizes as values.
            input_type: A dict with model input names as keys and the expected types as values.
            name: The model name according to Huggingface Transformers.
            device: A string that indicates where the model should run (cpu, cuda:0, etc...)
        """
        self.name = name
        """
        device = accelerator.device
        self._device = device
        if device is None:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(f"Model to {self._device}")
        """
        # self._pytorch_module = module.to(self._device).float().eval()
        # self._pytorch_module = module.float().eval()
        #self._pytorch_module = module.eval()

        #below up to print(f"[TorchModel] Model on {self._device}") is added to reduce the memory usage
        self._pytorch_module = module.eval()
        self._device = None
        if hasattr(module, "hf_device_map") and getattr(module, "hf_device_map"):
            # model is already placed across devices/CPU; leave as-is
            first_param = next(module.parameters(), None)
            self._device = first_param.device if first_param is not None else torch.device("cpu")
        else:
            # single-device placement
            if device is None:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._device = torch.device(device)
            self._pytorch_module = self._pytorch_module.to(self._device)
        print(f"[TorchModel] Model on {self._device}")

        if set(input_size.keys()) != set(input_type.keys()):
            raise RuntimeError(
                "Model input keys for size and type must be the same."
                f"{input_size.keys()} != {input_type.keys()}."
            )

        self._forward_hooks: t.List[RemovableHandle] = []
        self._input_size: t.Mapping[str, t.Tuple] = input_size
        self._input_types: t.Mapping[str, torch.dtype] = input_type
        self._response_infos: t.List[ResponseInfo] = []
        self._compute_response_infos()

    @property
    def module(self) -> nn.Module:
        return self._pytorch_module

    def _compute_response_infos(self) -> None:
        def hook(module_name, module, module_input, module_output) -> None:
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            outputs = module_output if isinstance(module_output, (list, tuple)) else [module_output]

            for output_idx, o in enumerate(outputs):
                if o is None or type(o) is not torch.Tensor:
                    continue

                response_name = "{}:{}".format(module_name, output_idx)
                ri = ResponseInfo(
                    name=response_name,
                    dtype=o.dtype,
                    shape=(o.size())[1:],
                    layer=ResponseInfo.Layer(
                        name=module_name,
                        kind=class_name,
                    ),
                )

                self._response_infos.append(ri)

        # register forward hook for all modules in the network with the exception of the root
        # module and container modules.
        hooks = []
        for module_name, module in self._pytorch_module.named_modules():
            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
                continue

            if module == self._pytorch_module:
                continue

            hooks.append(module.register_forward_hook(partial(hook, module_name)))

        # perform inference
        self._perform_dummy_inference()

        # remove forward hooks
        for h in hooks:
            h.remove()

    # to reduce memory usage, the datatype should be the same for dummy inference as well.
    def _perform_dummy_inference(self) -> None:
        family = transformers_model_name_to_family(self.name)

        def to_dev(x):  # prefer model's real device
            return x.to(next(self._pytorch_module.parameters()).device)

        # NEW: use model's parameter dtype for all float tensors
        first_param = next(self._pytorch_module.parameters(), None)
        param_dtype = first_param.dtype if first_param is not None else torch.float32

        B = 1  # was 2; keep dummy passes small
        if family != "seamlessm4t":
            x = {}
            for k, shape in self._input_size.items():
                dt = self._input_types[k]
                x[k] = to_dev(torch.rand((B, *shape)).type(dt))
            with torch.no_grad():
                self._pytorch_module(**x)
            return

        # -------- SeamlessM4T dummy discovery --------
        cfg = getattr(self._pytorch_module, "config", None)
        hidden_size = getattr(cfg, "hidden_size", 1024)
        dec_start = getattr(cfg, "decoder_start_token_id", None)
        if dec_start is None:
            dec_start = getattr(cfg, "bos_token_id", 1)

        # 1) TEXT ENCODER
        seq_len = self._input_size["input_ids"][0]
        input_ids = torch.randint(0, 256102, (B, seq_len), dtype=torch.long)
        attn_mask = torch.ones(B, seq_len, dtype=torch.long)
        with torch.no_grad():
            _ = self._pytorch_module.text_encoder(
                input_ids=to_dev(input_ids),
                attention_mask=to_dev(attn_mask),
            )

        # 2) SPEECH ENCODER  (make float tensors in model dtype)
        T = self._input_size["input_features"][0]  # e.g., 300
        F = self._input_size["input_features"][1]  # e.g., 160
        feats = torch.randn(B, T, F, dtype=param_dtype)        # CHANGED
        speech_mask = torch.ones(B, T, dtype=torch.long)
        with torch.no_grad():
            _ = self._pytorch_module.speech_encoder(
                input_features=to_dev(feats),
                attention_mask=to_dev(speech_mask),
            )

        # 3) DECODER (encoder states in model dtype)
        dec_len = 3
        decoder_input_ids = torch.full((B, dec_len), int(dec_start), dtype=torch.long)
        decoder_attention_mask = torch.ones(B, dec_len, dtype=torch.long)
        enc_T = 8
        enc_states = torch.randn(B, enc_T, hidden_size, dtype=param_dtype)  # CHANGED
        enc_mask = torch.ones(B, enc_T, dtype=torch.long)
        with torch.no_grad():
            try:
                _ = self._pytorch_module.text_decoder(
                    input_ids=to_dev(decoder_input_ids),
                    attention_mask=to_dev(decoder_attention_mask),
                    encoder_hidden_states=to_dev(enc_states),       # CHANGED dtype-compatible
                    encoder_attention_mask=to_dev(enc_mask),
                    use_cache=False,
                )
            except TypeError:
                _ = self._pytorch_module.text_decoder(
                    input_ids=to_dev(decoder_input_ids),
                    encoder_hidden_states=to_dev(enc_states),
                    use_cache=False,
                )

    def get_response_infos(self) -> t.Iterable[ResponseInfo]:
        """
        Generate a list of :class:`ResponseInfo`s with the name, type and other information of each response.
        Returns:
            A list of :class:`ResponseInfo` objects.
        """
        return self._response_infos
    
    #modified so that it can deal with transposed shape in recording activations
    def _set_units_hook_wrapper(
        self,
        units: torch.Tensor,
        values: torch.Tensor,
        only_last_token: bool,
        layer_name: str = None,
    ) -> t.Callable:
        assert len(units) == len(values), "The number of values must match the number of units."

        def forward_hook(module, input, output):
            if layer_name and "self_attn.distance_embedding" in layer_name:
                return output

            out = output
            transposed = False
            if out.dim() == 3 and out.shape[1] > out.shape[2]:
                out = out.transpose(1, 2)  # (B, H, T) -> (B, T, H)
                transposed = True

            squeezed_2d = False
            if out.dim() == 2:
                out = out.unsqueeze(1)  # (B, H) -> (B, 1, H)
                squeezed_2d = True

            vals = values.to(device=out.device, dtype=out.dtype)

            # Write only if we now have (B, T, H)
            if out.dim() == 3:
                if only_last_token:
                    out[:, -1, units] = vals
                else:
                    out[:, :, units] = vals
            # else: silently skip (no safe feature axis to write to)

            if squeezed_2d:
                out = out.squeeze(1)
            if transposed:
                out = out.transpose(1, 2)  # restore original

            return out

        return forward_hook
    
    def set_units_in_layer(
        self,
        layer_name: str,
        units: torch.Tensor,
        values: torch.Tensor,
        only_last_token: bool = False,
    ) -> None:
        layer_name = layer_name.replace(":0", "")

        # Skip distance embedding layers entirely (no intervention)
        if "self_attn.distance_embedding" in layer_name:
            return

        for iterated_module_name, layer in self._pytorch_module.named_modules():
            if iterated_module_name == layer_name:
                handle = layer.register_forward_hook(
                    self._set_units_hook_wrapper(
                        units=units,
                        values=values,
                        only_last_token=only_last_token,
                        layer_name=layer_name,  # pass through for the hook’s check
                    )
                )
                self._forward_hooks.append(handle)

    
    def restore_units(self):
        for h in self._forward_hooks:
            h.remove()
        self._forward_hooks.clear()
    
    #the function below was modified in order to adapt to the names that seamless expects
    def run_inference(self, inputs: t.Mapping[str, torch.Tensor], outputs: t.AbstractSet[str]) -> t.Dict[str, np.ndarray]:
        a_key = list(inputs.keys())[0]
        torch_inputs: t.MutableMapping[str, torch.Tensor] = {}

        # move inputs to the same device as model
        #device = next(self._pytorch_module.parameters()).device
        #for k, v in inputs.items():
        #    torch_inputs[k] = v.to(device)

        # move inputs to the same device as model
        #changed because input ids should be long and not bf16 (ok for speech features)
        device = next(self._pytorch_module.parameters()).device
        dtype = getattr(self, "_torch_dtype", None)  # あなたの実装に合わせて

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k in ("input_ids", "decoder_input_ids", "labels") or k.endswith("_input_ids"):
                    torch_inputs[k] = v.to(device=device, dtype=torch.long)
                elif v.is_floating_point():
                    torch_inputs[k] = v.to(device=device, dtype=(dtype or v.dtype))
                else:
                    torch_inputs[k] = v.to(device=device)
            else:
                torch_inputs[k] = v
                
        # --- SeamlessM4T normalization: ensure decoder_* kwargs exist and are correctly named ---
        is_seamless = hasattr(self._pytorch_module, "text_decoder") and hasattr(self._pytorch_module, "text_encoder")
        if is_seamless:
            # Common aliases -> official names
            alias_map = {
                "dec_input_ids": "decoder_input_ids",
                "decoder_ids": "decoder_input_ids",
                "tgt_input_ids": "decoder_input_ids",
                "decoder_mask": "decoder_attention_mask",
                "dec_attention_mask": "decoder_attention_mask",
                "tgt_attention_mask": "decoder_attention_mask",
            }
            for old, new in alias_map.items():
                if old in torch_inputs and new not in torch_inputs:
                    torch_inputs[new] = torch_inputs.pop(old)

            # Some pipelines use labels as decoder inputs for teacher forcing
            if "decoder_input_ids" not in torch_inputs and "labels" in torch_inputs:
                torch_inputs["decoder_input_ids"] = torch_inputs["labels"]

            # If still missing, create a minimal 1-token BOS decoder prompt
            if "decoder_input_ids" not in torch_inputs:
                B = next(iter(torch_inputs.values())).size(0)
                bos = getattr(self._pytorch_module.config, "decoder_start_token_id", 3)  # Seamless uses 3 typically
                torch_inputs["decoder_input_ids"] = torch.full((B, 1), bos, dtype=torch.long, device=device)
                torch_inputs.setdefault("decoder_attention_mask",
                    torch.ones(B, 1, dtype=torch.long, device=device)
                )

            # If attention mask missing, make a full-one mask matching length
            if "decoder_attention_mask" not in torch_inputs:
                di = torch_inputs["decoder_input_ids"]
                torch_inputs["decoder_attention_mask"] = (di != 0).long()
            '''
            #debug用
            #print({k: (tuple(v.shape), v.dtype) for k, v in torch_inputs.items()})

            from transformers import SeamlessM4TProcessor

            if not hasattr(self, "_dbg_processor"):
                self._dbg_processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

            tok = self._dbg_processor.tokenizer

            # 例: decoder_input_ids をdecode
            #di = torch_inputs["decoder_input_ids"].detach().cpu().tolist()
            #print("decoder ids[0]:", di[0])
            #print("decoded[0]:", tok.decode(di[0], skip_special_tokens=False))
            dec_ids = torch_inputs.get("decoder_input_ids", None)
            if dec_ids is None:
                print("[DEBUG] no decoder_input_ids. keys:", list(torch_inputs.keys()))
            else:
                ids0 = dec_ids[0].detach().to("cpu").to(torch.long).tolist()
                print("decoded[0]:", tok.decode(ids0, skip_special_tokens=False))
                #ここまでdebug用で、あとで消す
            '''

        # --------- collect hook outputs as you already do (unchanged) ----------
        response_dict: t.Dict[str, t.Any] = {}
        def hook(module_name, module, module_input, module_output):
            outs = module_output if isinstance(module_output, (list, tuple)) else [module_output]
            for output_idx, o in enumerate(outs):
                rn = f"{module_name}:{output_idx}"
                if rn in outputs:
                    if o.dtype == torch.float32:
                        tensor = o.detach().cpu().numpy()
                    else:
                        tensor = o.detach().to(dtype=torch.float32).cpu().numpy()
                    response_dict[rn] = tensor

        hooks = []
        for module_name, module in self._pytorch_module.named_modules():
            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):  # skip containers
                continue
            if module is self._pytorch_module:
                continue
            hooks.append(module.register_forward_hook(partial(hook, module_name)))

        with torch.no_grad():
            self._pytorch_module(**torch_inputs)

        for h in hooks:
            h.remove()

        return response_dict

class PytorchTransformersModel(TorchModel):
    """
    Class wrapping a HuggingFace Transformers model in a readable model.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: t.Optional[pathlib.Path],
        seq_len: int,
        device: str = None,
        #clm: str = False,
        #bit: str = False,
    ) -> None:
        """
        Loads a HuggingFace Transformers given its name.

        Args:
            model_name: The model name
            cache_dir: Local dir where the model is fetched/saved
            seq_len: Input sequence length considered.
        """
        torch_model = transformers_class_from_name(model_name, cache_dir=cache_dir) #, clm=clm, bit=bit)
        #torch_model = accelerator.prepare(torch_model)
        input_size, input_type = _input_spec_for_family(model_name, seq_len) #added
        super().__init__(
            module=torch_model,
            input_size=input_size, #{input_name: (seq_len,) for input_name in MODEL_INPUT_FIELDS},
            input_type=input_type, #{input_name: torch.long for input_name in MODEL_INPUT_FIELDS},
            name=model_name,
            device=device,
        )


def transformers_model_name_to_family(model_name: str) -> str:
    """
    Get the family of the model based on the model name, as defined in the Huggingface transformers repository.

    For example: `bert-base-cased` belongs to the family `bert`
    Args:
        model_name: The model name

    Returns:
        str: The family name

    """
    if model_name.startswith("bert"):
        return "bert"
    elif model_name.startswith("openai"):
        return "openai"
    elif model_name.startswith("gpt2"):
        return "gpt2"
    elif model_name.startswith("xlnet"):
        return "xlnet"
    elif model_name.startswith("xlm"):
        return "xlm"
    elif model_name.startswith("roberta"):
        return "roberta"
    elif model_name.startswith("distilbert"):
        return "distilbert"
    elif model_name.startswith("ctrl"):
        return "ctrl"
    elif "bloom" in model_name:
        return "bloom"
    elif "Llama-2" in model_name:
        return "Llama-2"
    elif "llama" in model_name:
        return "llama"
    elif "falcon" in model_name:
        return "falcon"
    elif "xglm" in model_name:
        return "xglm"
    elif ("seamless" in model_name and "m4t" in model_name) or "seamlessm4t" in model_name:
        return "seamlessm4t"
    else:
        raise NotImplementedError(f"Model name to type not considered: {model_name}")

#modified to make it more memory friendly
def transformers_class_from_name(
    model_name: str, cache_dir: t.Optional[pathlib.Path] = None, rand_weights: bool = False
) -> nn.Module:
    """
    Obtain a model as pytorch nn.Module given a name (as defined in the Huggingface transformers repo)
    """
    try:
        if rand_weights:
            config = AutoConfig.from_pretrained(model_name)
            m = AutoModelForPreTraining.from_config(config)
            return m

        # Common, memory-friendly kwargs
        common_kwargs = {
            "cache_dir": cache_dir,
            "low_cpu_mem_usage": True,
            "device_map": "auto",               # shard across GPU/CPU
        }

        torch_dtype = _select_torch_dtype()
        # Only apply dtype when it makes sense (fp16/bf16 on CUDA; fp32 on CPU is fine)
        if torch_dtype in (torch.float16, torch.bfloat16):
            common_kwargs["torch_dtype"] = torch_dtype

        # Optional 8-bit loading
        load_in_8bit = _want_8bit() and _bitsandbytes_available()
        if load_in_8bit:
            # If we request 8-bit, dtype arg should not be passed
            common_kwargs.pop("torch_dtype", None)
            common_kwargs["load_in_8bit"] = True

        if 'Llama-2' in model_name or 'llama' in model_name.lower():
            m = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **common_kwargs
            )
        elif 'xglm' in model_name:
            m = XGLMForCausalLM.from_pretrained(
                model_name,
                **common_kwargs
            )
        elif 'bloom' in model_name:
            m = BloomForCausalLM.from_pretrained(
                model_name,
                **common_kwargs
            )
        elif ("seamless" in model_name.lower() and "m4t" in model_name.lower()) or "seamlessm4t" in model_name.lower() or "SeamlessM4T" in model_name:
            # Encoder-decoder speech/text model
            m = SeamlessM4Tv2Model.from_pretrained(
                model_name,
                **common_kwargs
            )
        else:
            # Fall back to a generic pretraining head (covers BERT-like, etc., if you add them later)
            m = AutoModelForPreTraining.from_pretrained(
                model_name,
                **common_kwargs
            )

        try:
            print(getattr(m, "hf_device_map", None))
        except Exception:
            pass

    except OSError as e:
        raise NotImplementedError(f"Model {model_name} could not be loaded. {e}")
    assert m is not None
    return m

#newly added
def _input_spec_for_family(model_name: str, seq_len: int, speech_T: int = 300):
    family = transformers_model_name_to_family(model_name)
    if family in {"gpt2", "bloom", "llama", "Llama-2", "xglm"}:
        input_size = {k: (seq_len,) for k in MODEL_INPUT_FIELDS}
        input_types = {k: torch.long for k in MODEL_INPUT_FIELDS}
        return input_size, input_types

    if family == "seamlessm4t":
        # We will prepare BOTH text and speech specs so we can run two dummy passes.
        input_size = {
            # text path
            "input_ids": (seq_len,),
            "attention_mask": (seq_len,),
            # speech path (log-mel features): (time, 160) -> match our feature_projection (maybe to change)
            "input_features": (speech_T, 160),
        }
        input_types = {
            "input_ids": torch.long,
            "attention_mask": torch.long,
            "input_features": torch.float32,
        }
        return input_size, input_types

    raise NotImplementedError(f"No input spec for family {family}")


def get_layer_regex(model_name: str) -> t.Optional[t.List[str]]:
    """
    Create regex for the layers of interest for different model families.
    These are the layers where expert units will be explored.

    Note:
        Only GPT2 family supported for now.

    Args:
        model_name: The requested model name.

    Returns:
        A list of strings with the layer names.

    """
    family = transformers_model_name_to_family(model_name)
    layer_types = None
    if family == "gpt2":
        layer_types = [
            "transformer.h.([0-9]|[0-9][0-9]).attn.c_attn",
            "transformer.h.([0-9]|[0-9][0-9]).attn.c_proj",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.c_fc",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.c_proj",
        ]
    elif family == "bloom":
        layer_types = [
            "transformer.h.([0-9]|[0-9][0-9]).self_attention.query_key_value",
            "transformer.h.([0-9]|[0-9][0-9]).self_attention.dense",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.dense_h_to_4h",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.dense_4h_to_h",
        ]
    elif family in ["llama", "Llama-2"]:
        layer_types = [
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.q_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.k_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.v_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.o_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.gate_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.down_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.up_proj",
        ]
    elif family == "xglm":
        layer_types = [
            "model.layers.([0-9]|[0-9][0-9]).self_attn.k_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.v_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.q_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.out_proj",
            "model.layers.([0-9]|[0-9][0-9]).fc1",
            "model.layers.([0-9]|[0-9][0-9]).fc2",
        ]
    elif family == "seamlessm4t":
        # Text encoder projections & FFN
        layer_types = [
            r"text_encoder\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
            r"text_encoder\.layers\.[0-9]+\.ffn\.(fc1|fc2)",
            # Speech encoder: Conformer self-attn and conv module + FFNs
            r"speech_encoder\.encoder\.layers\.[0-9]+\.self_attn\.(linear_q|linear_k|linear_v|linear_out)",
            r"speech_encoder\.encoder\.layers\.[0-9]+\.conv_module\.(pointwise_conv1|depthwise_conv|pointwise_conv2)",
            r"speech_encoder\.encoder\.layers\.[0-9]+\.ffn1\.(intermediate_dense|output_dense)",
            r"speech_encoder\.encoder\.layers\.[0-9]+\.ffn2\.(intermediate_dense|output_dense)",
            # Decoder: self-attn, cross-attn, FFN
            r"text_decoder\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
            r"text_decoder\.layers\.[0-9]+\.cross_attention\.(q_proj|k_proj|v_proj|out_proj)",
            r"text_decoder\.layers\.[0-9]+\.ffn\.(fc1|fc2)",
        ]
    # Extend to other model families here if needed
    return layer_types


def _print_responses(ri: t.List[ResponseInfo]) -> None:
    assert len(ri), "No responses selected"
    print(f"Found {len(ri)} responses from model.")
    for r in ri:
        print("\t", r.name, r.shape)


def _collect_responses_info_for_model(model: TorchModel, model_family: str) -> t.List[ResponseInfo]:
    mapping = {
        "gpt2": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "bloom": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "llama": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "Llama-2": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "falcon": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "xglm": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "seamlessm4t": [
            ri for ri in model.get_response_infos()
            if ri.layer.kind in ["Linear", "Conv1d", "LayerNorm", "Embedding", "GLU"]  # generous
            and len(ri.shape) in [2, 3]
            and (
                ri.layer.name.startswith("text_encoder.") or
                ri.layer.name.startswith("speech_encoder.") or
                ri.layer.name.startswith("text_decoder.")
            )
        ],
        #LayerNorm, embedding, GLU can be dropped. think more and choose
        # Extend to other models here
    }
    return mapping[model_family]


def collect_responses_info(model_name: str, model: TorchModel) -> t.List[ResponseInfo]:
    """
    Build the information required to read responses from model.

    Args:
        model_name: The model name
        model: A TorchModel

    Returns:
        Responses info

    """
    family = transformers_model_name_to_family(model_name)
    responses_info = _collect_responses_info_for_model(model, family)
    _print_responses(responses_info)
    return responses_info


def concatenate_responses(
    responses: t.Dict[str, np.ndarray],
    response_fields: t.Set[str],
    output_field: str,
    axis: int,
) -> t.Dict[str, np.ndarray]:
    data = [tensor for field, tensor in responses.items() if field in response_fields]
    responses[output_field] = np.concatenate(data, axis=axis)
    for field in response_fields:
        del responses[field]
    return responses

#rewritten because some conv in speech encoder has swapped dimensions, like hid_dim at the beginning
#however, the function below only works for Batch=1 as it assumes ndim = 2.
def pool_responses(
    responses: Dict[str, np.ndarray],
    response_fields: Optional[Set[str]],
    axis: Union[int, Tuple[int, ...]] = 1,
    pooling_type: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    1) For any response with shape (A, B) (i.e., 2-D), *except* keys containing
       'self_attn.distance_embedding', if A > B, transpose to (B, A).
    2) Then apply the chosen pooling over `axis` for the selected fields.

    Args
    ----
    responses : dict[str, np.ndarray]
        Hooked activations per layer name.
    response_fields : set[str] | None
        Which keys to pool. None => pool all keys in `responses`.
    axis : int | tuple[int, ...]
        Axis/axes to reduce with the pooling op (after any transpose).
    pooling_type : {'mean','sum','max','median','min'}
        Pooling operation.

    Returns
    -------
    dict[str, np.ndarray]
        Same dict, with selected arrays pooled (and possibly transposed beforehand).
    """
    assert pooling_type in ["mean", "sum", "max", "median", "min"]
    pooler_fn = getattr(np, pooling_type)
    fields = response_fields or responses.keys()

    for key in fields:
        arr = responses[key]
        # Ensure numpy array
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        #print("debug:", key)
        #transpose for conv 
        if (
            key.startswith("speech_encoder.")
            and arr.ndim == 3
            and "self_attn.distance_embedding" not in key
            and arr.shape[1] > arr.shape[2]
        ):
            arr = np.transpose(arr, (0, 2, 1))  # (A,B) -> (B,A)

        # --- (2) Pool along axis ---
        #print(f"debug info1: {key} - {arr.shape}")
        responses[key] = pooler_fn(arr, axis=axis)
        #print(f"debug info2: {key} - {responses[key].shape}")

    return responses



def processors_per_model(model: TorchModel) -> t.List[t.Callable]:
    #pool_args: t.List[t.Dict] = [dict(response_fields=None, axis=1, pooling_type="min")]
    pool_args: t.List[t.Dict] = [dict(response_fields=None, axis=1, pooling_type="mean")]
    #pool_args: t.List[t.Dict] = [dict(response_fields=None, axis=1, pooling_type="median")]
    #pool_args: t.List[t.Dict] = [dict(response_fields=None, axis=1, pooling_type="max")]
    process_fns: t.List[t.Callable] = []
    process_fns += [partial(pool_responses, **args) for args in pool_args]
    return process_fns
