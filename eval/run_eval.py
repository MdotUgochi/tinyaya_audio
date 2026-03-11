import sys
sys.path.insert(0, '/content/tinyaya_audio/src')import argparse
import os
import torch
import numpy as np
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoTokenizer, WhisperFeatureExtractor, Cohere2Config
from transformers.models.tinyaya_audio import (
    TinyAyaAudioForConditionalGeneration,
    TinyAyaAudioProcessor,
    TinyAyaAudioConfig,
)
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')

def main(args):
    text_cfg = Cohere2Config(
        hidden_size=2048,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=11008,
    )
    config = TinyAyaAudioConfig(
        text_config=text_cfg,
        audio_token_index=261002,
    )

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    model = TinyAyaAudioForConditionalGeneration(config)
    model = model.to(torch.bfloat16).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "C:/Users/USER/Desktop/ListAya/transformers/tinyaya_audio_tokenizer"
    )
    feature_extractor = WhisperFeatureExtractor()
    processor = TinyAyaAudioProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    gen_kwargs = {"max_new_tokens": args.max_new_tokens or 200}



    def benchmark(batch, min_new_tokens=None):
        audios = [audio["array"].astype(np.float32) for audio in batch["audio"]]
        minibatch_size = len(audios)
        batch["audio_length_s"] = [len(audio) / 16_000 for audio in audios]

        start_time = time.time()

        #TinyAya processor handles one audio at a time 
        pred_ids_list = []
        for audio in audios:
            prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech."
            inputs = processor(
                text=prompt,
                audio=audio,
                sampling_rate=16000,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

            with torch.no_grad():
                pred_ids = model.generate(**inputs, **gen_kwargs)
            pred_ids_list.append(pred_ids[0])
            
            runtime = time.time() - start_time

        pred_text = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in pred_ids_list
        ]

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch   

       






    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
    )
    parser.add_argument(
        "--longform",
        action="store_true",
        help="Whether to use longform mode.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to JIT compile the forward pass of the model.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        help="Mode for torch compiling model forward pass. Can be either 'default', 'reduce-overhead', 'max-autotune' or 'max-autotune-no-cudagraphs'.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
