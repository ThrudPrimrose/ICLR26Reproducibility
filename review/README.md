# Review plots (not in the paper)

`ml16-{strong,weak}.png`: all 20 distributed ML operators graded on 1-16 GPUs by the multi-node
scaling grade (`mirror/mlscale-grade/`: part 1, part 2 and the Kimi runs of each), as speedup over
PyTorch on one GPU; the grey dashed curve is `torch_dist`, PyTorch's own distributed reference.
Each model's two prompts are pooled per operator (better curve kept). CSVs hold every point.

    lib/torch_anchor.py grade16.db grade16-torch.db --best-of <gemmhint>=<base> ...
    statistics/plot_scaling.py grade16-torch.db --experiment mlscale --figure per-kernel \
        --mode strong --quantity speedup --out review/ml16-strong
