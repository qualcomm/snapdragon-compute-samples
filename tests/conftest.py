from pytest import fixture
from pathlib import Path
import json

@fixture
def root_dir():
    return Path.cwd()

@fixture
def model_paths(root_dir):
    model_config_data = {"MODELS": {
        "HRNET_POSE": {
            "PATH_SUBDIRECTORY": "hrnet_pose",
            "DEFAULT": "hrnet_pose.onnx",
            "QUANTIZED": "hrnet_quantized.onnx"
            },
        "DEEPSEEK_7B": {
            "PATH_SUBDIRECTORY": "qnn-deepseek-r1-distill-qwen-7b",
            "DEFAULT":{
                "EMBEDDING": "deepseek_r1_7b_embeddings_quant_v1.0.onnx",
                "CONTEXT": "deepseek_r1_7b_ctx_v1.0.onnx_ctx.onnx",
                "CONTEXT_ITER": "deepseek_r1_7b_iter_v1.0.onnx_ctx.onnx",
                "HEAD": "deepseek_r1_7b_head_quant_v1.0.onnx",
                "TOKENIZER": "tokenizer.json",
                "META_DATA": {"num_heads": 28,
                            "num_key_value_heads": 4,
                            "num_layers": 28,
                            "attn_head_size": 128,
                            "max_seq_len": 64}
                }
            },
        "DEEPSEEK_1.5B": {
            "PATH_SUBDIRECTORY": "qnn-deepseek-r1-distill-qwen-1.5b",
            "DEFAULT":{
                "EMBEDDING": "deepseek_r1_1_5_embeddings_quant_v2.0.onnx",
                "CONTEXT": "deepseek_r1_1_5_ctx_v2.1.onnx_ctx.onnx",
                "CONTEXT_ITER": "deepseek_r1_1_5_iter_v2.1.onnx_ctx.onnx",
                "HEAD": "deepseek_r1_1_5_head_quant_v2.0.onnx",
                "TOKENIZER": "tokenizer.json",
                "META_DATA": {"num_heads": 12,
                            "num_key_value_heads": 2,
                            "num_layers": 28,
                            "attn_head_size": 128,
                            "max_seq_len": 64}
                }
            }
        }
    }

    def _get_path(model: str):
        
        model_path_sub = model_config_data["MODELS"][model].get("PATH_SUBDIRECTORY")
        
        return Path.joinpath(root_dir,"models",model_path_sub)
    return _get_path

@fixture
def load_config(root_dir):

    def _get_config(config: str):
        model_json = Path.joinpath(root_dir, config)
        with open(model_json, "r") as f:
            model_config = json.load(f)
        return model_config

    return _get_config 