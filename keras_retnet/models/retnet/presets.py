# pylint:disable=line-too-long
backbone_presets = {
    "retnet-base": {
        "metadata": {
            "description": "base 6-layer retnet architecture",
            # "params": 169_342_464,
            "official_name": "RetNet",
            "path": "retnet",
            # "model_card": "https://huggingface.co/BlinkDL/rwkv-4-pile-169m",
        },
        "config": {
            "hidden_dim": 512,
            "intermediate_dim": 1024,
            # "intermediate_dim": 2048,
            "num_layers": 6,
            "num_heads": 2,
            "activation": "gelu",
            # "vocabulary_size": 267_744,
            "vocabulary_size": 50_257,
        },
        # "max_sequence_length": 1024,
        # "weights_url": "https://huggingface.co/parsee-mizuhashi/retnet/resolve/main/retnet-1m.safetensors",
        # "weights_hash": "7a0b16a33bf029a12cc7e39df116fef329f2e659b20f73b0c2c6d6a43985e829",
    },
    "retnet-medium": {
        "metadata": {
            "description": "16 layer retnet",
            "official_name": "RetNet",
            "path": "retnet",
        },
        "config": {
            "hidden_dim": 1024,
            "intermediate_dim": 2048,
            "num_layers": 16,
            "num_heads": 4,
        },
    },
    "retnet-xl": {
        "metadata": {
            "description": "24 layer retnet",
            "official_name": "RetNet",
            "path": "retnet",
        },
        "config": {
            "hidden_dim": 2048,
            "intermediate_dim": 4096,
            "num_layers": 24,
            "num_heads": 8,
        },
    },
    "retnet-3b": {
        "metadata": {
            "description": "3 billion parameter retnet",
            "official_name": "RetNet",
            "path": "retnet",
        },
        "config": {
            "hidden_dim": 2560,
            "intermediate_dim": 5120,
            "num_layers": 32,
            "num_heads": 10,
        },
    },
    "retnet-7b": {
        "metadata": {
            "description": "7 billion parameter retnet",
            "official_name": "RetNet",
            "path": "retnet",
        },
        "config": {
            "hidden_dim": 4096,
            "intermediate_dim": 8192,
            "num_layers": 32,
            "num_heads": 16,
        },
    },
    "retnet-13b": {
        "metadata": {
            "description": "13 billion parameter retnet",
            "official_name": "RetNet",
            "path": "retnet",
        },
        "config": {
            "hidden_dim": 5120,
            "intermediate_dim": 10240,
            "num_layers": 40,
            "num_heads": 20,
        },
    },
    "retnet-65b": {
        "metadata": {
            "description": "65 billion parameter retnet",
            "official_name": "RetNet",
            "path": "retnet",
        },
        "config": {
            "hidden_dim": 8192,
            "intermediate_dim": 16384,
            "num_layers": 64,
            "num_heads": 32,
        },
    },
}
