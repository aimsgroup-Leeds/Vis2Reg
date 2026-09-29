"""Inference contract: GeometryInput(P, merged_Q) -> RegistrationOutput."""


def main() -> None:
    # Load checkpoint; model.eval(); inference_mode(); model(geometry, deform=True).
    # Masks and K may construct Q upstream; never pass them to the registration model.
    # Require metric scale and explicit multiview coordinate alignment upstream.
    raise NotImplementedError("Bind geometric inputs, model adapters and checkpoint.")


if __name__ == "__main__":
    main()
