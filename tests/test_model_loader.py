
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions in LICENSE.txt are met

from unittest.mock import patch
from src.model_loader import ModelLoader


def test_model_loader_initialization(model_paths):
      
    loader = ModelLoader(model="deepseek_7b", processor="npu", model_type="default")
    dll_path = str(loader._get_dll_path())
    assert loader.model == "deepseek_7b"
    assert loader.processor == "NPU" if loader.arch=="arm64" else "CPU" 
    assert loader.model_type == "default"
    assert "EP" in loader._get_executioner()
    assert dll_path.endswith(".dll")
    assert loader.graphs["EMBEDDING"].endswith(".onnx")

    expected_path = model_paths("DEEPSEEK_7B")
    assert str(loader.model_subdirectory_path) == str(expected_path)

#Inject x86 into system architecture if ARM64 to check for CPU fallback
@patch("platform.machine", return_value="x86_64")
@patch("platform.system", return_value="Windows")
def test_processor_fallback(mock_system, mock_machine):
    loader = ModelLoader(model="deepseek_7B", processor="npu", model_type="quantized")
    assert loader.processor == "CPU"  # fallback triggered inside platform_manager()

def test_validate_model_config_structure(load_config):
    config = load_config("models.json")

    assert "MODELS" in config, "Missing 'MODELS' key"
    assert isinstance(config["MODELS"], dict), "'MODELS' must be a dictionary"

    for model_name, model_data in config["MODELS"].items():
        assert "PATH_SUBDIRECTORY" in model_data, f"{model_name} missing 'PATH_SUBDIRECTORY'"
        assert isinstance(model_data["PATH_SUBDIRECTORY"], str), f"{model_name} 'PATH_SUBDIRECTORY' must be a string"

        default = model_data.get("DEFAULT")
        assert default is not None, f"{model_name} missing 'DEFAULT'"

        # Handle string or dict form
        if isinstance(default, str):
            assert default.endswith(".onnx"), f"{model_name} 'DEFAULT' must be an ONNX filename"
        elif isinstance(default, dict):
            required_keys = ["EMBEDDING", "CONTEXT", "CONTEXT_ITER", "HEAD", "TOKENIZER", "META_DATA"]
            for key in required_keys:
                assert key in default, f"{model_name} DEFAULT missing '{key}'"
            assert isinstance(default["META_DATA"], dict), f"{model_name} 'META_DATA' must be a dictionary"
        else:
            raise AssertionError(f"{model_name} 'DEFAULT' must be a string or dictionary")

def test_validate_executioner_config_structure(load_config):
    config = load_config("executioner.json")

    assert isinstance(config, dict), "Config should be a dictionary"

    for processor, entry in config.items():
        assert isinstance(entry, dict), f"{processor} config must be a dictionary"
        assert "EP" in entry, f"{processor} config missing 'EP'"
        assert isinstance(entry["EP"], str), f"{processor} 'EP' must be a string"
        assert "PATH" in entry, f"{processor} config missing 'PATH'"
        assert isinstance(entry["PATH"], str), f"{processor} 'PATH' must be a string"
        assert entry["PATH"].endswith(".dll"), f"{processor} 'PATH' must end with '.dll'"


                                           
