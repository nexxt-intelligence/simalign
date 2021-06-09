from os import environ
from psutil import cpu_count

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
environ["OMP_WAIT_POLICY"] = "ACTIVE"

from contextlib import contextmanager
from dataclasses import dataclass
from time import time

from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_all_providers,
)
from tqdm import trange
from transformers import BertTokenizerFast
import retrieve
from pathlib import Path
import numpy as np


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    assert (
        provider in get_all_providers()
    ), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session


class MBERTOnnx:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-multilingual-cased"
        )
        model_path = retrieve.url(
            "https://github.com/nexxt-intelligence/simalign/releases/download/mbert-quantized/bert-base-multilingual-cased-quantized.onnx"
        )
        quantized_model_path = Path(model_path)
        self.model = create_model_for_provider(
            quantized_model_path.as_posix(), "CPUExecutionProvider"
        )

    def predict(self, text):
        if not isinstance(text[0], str):
            model_inputs = self.tokenizer(
                text,
                is_split_into_words=True,
                padding=True,
                truncation=True
            )
        else:
            model_inputs = self.tokenizer(
                text,
                is_split_into_words=False,
                padding=True,
                truncation=True
            )
        inputs_onnx = {k: np.array(v) for k, v in model_inputs.items()}
        outputs = self.model.run(None, inputs_onnx)
        # Get hidden states
        pooled = outputs[2:]
        # Get token outputs for 8th layer
        outputs = pooled[8]
        # Skip [CLS] token embedding
        return outputs[:, 1:-1, :]
