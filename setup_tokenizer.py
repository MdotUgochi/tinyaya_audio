from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-base")

tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "<|AUDIO|>",
        "<|audio_bos|>",
        "<|audio_eos|>"
    ]
})

print("audio token id:    ", tokenizer.convert_tokens_to_ids("<|AUDIO|>"))
print("audio_bos token id:", tokenizer.convert_tokens_to_ids("<|audio_bos|>"))
print("audio_eos token id:", tokenizer.convert_tokens_to_ids("<|audio_eos|>"))
print("vocab size now:    ", len(tokenizer))

tokenizer.save_pretrained("./tinyaya_audio_tokenizer")
print("Tokenizer saved to ./tinyaya_audio_tokenizer")