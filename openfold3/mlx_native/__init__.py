"""Pure-MLX inference port of OpenFold3.

Reimplements the OpenFold3 forward pass in Apple MLX (no torch in the hot path),
loading the released torch checkpoint into MLX arrays. Built bottom-up with a
numerical parity gate against the torch reference at every stage.
"""
