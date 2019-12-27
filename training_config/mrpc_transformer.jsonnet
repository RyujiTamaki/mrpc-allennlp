local model_name = "bert-base-uncased";

{
    "dataset_reader": {
        "lazy": false,
        "type": "mrpc",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
            }
        },
    },
    "train_data_path": std.extVar("MRPC_TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("MRPC_VALID_DATA_PATH"),
    "model": {
        "type": "basic_classifier",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "tokens": ["tokens", "attention_mask"],
            },
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": model_name,
                }
            }
        },
        "seq2vec_encoder": {
           "type": "cls_token",
           "hidden_size": 768,
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 2e-5
        },
        "validation_metric": "+accuracy",
        "num_epochs": 3,
        "cuda_device": 0
    }
}
