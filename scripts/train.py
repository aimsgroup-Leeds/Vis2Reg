"""Stage 2: pretrained rigid seed -> 20 rigid + 40 joint real-data epochs."""


def main() -> None:
    # Bind real dataset, model, VisibilityAwareObjective and Trainer per official fold.
    # AdamW 3e-4 / 1e-4, cosine, AMP, batch size 1; geometry-only model inputs.
    raise NotImplementedError("Bind official folds, checkpoint and training adapters.")


if __name__ == "__main__":
    main()
