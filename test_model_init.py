import torch
from transformers.models.tinyaya_audio import TinyAyaAudioConfig, TinyAyaAudioForConditionalGeneration
from transformers import AutoConfig

# Use a tiny Cohere2 config to avoid OOM on CPU
from transformers.models.cohere2 import Cohere2Config

text_cfg = Cohere2Config(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
)

config = TinyAyaAudioConfig(
    text_config=text_cfg,
    audio_token_index=261002,  # confirmed value
    audio_config={
        "model_type": "tinyaya_audio_encoder",
        "d_model": 64,
        "encoder_layers": 2,
        "encoder_attention_heads": 4,
        "encoder_ffn_dim": 128,
        "num_mel_bins": 128,
        "max_source_positions": 1500,
    }
)

model = TinyAyaAudioForConditionalGeneration(config)
print("Model created OK")
print("Projector weight shape:", model.multi_modal_projector.linear.weight.shape)
# Expected: torch.Size([256, 64])  — [text_hidden_size, audio_d_model]
print("LLM type:", type(model.language_model).__name__)
# Expected: Cohere2ForCausalLM