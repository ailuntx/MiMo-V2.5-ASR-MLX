import argparse

from mlx_audio.stt import load


def parse_args():
    parser = argparse.ArgumentParser(description="Run MiMo-V2.5-ASR with MLX-Audio.")
    parser.add_argument(
        "--model",
        default="mlx-community/MiMo-V2.5-ASR-MLX",
        help="Local model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint. Use en, zh, or leave unset for auto.",
    )
    parser.add_argument(
        "--audio-tokenizer-dir",
        default=None,
        help="Optional local MiMo-Audio-Tokenizer directory override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load(args.model, audio_tokenizer_dir=args.audio_tokenizer_dir)
    result = model.generate(args.audio, language=args.language)
    print(result.text)


if __name__ == "__main__":
    main()
