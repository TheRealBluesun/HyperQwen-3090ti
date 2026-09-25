"""qwen27-dev: M<=8 Marlin configs (thread_k, thread_n, CTA count) for Qwen3.8-27B W4A16 + DFlash2 on an
RTX 3090 @ 350 W. Swept thread configs x CTA counts 24..82 at M=8; default = stock heuristic (128,128)x82."""
_TABLE = {
    (16384, 5120): (64, 128, 64),    # GDN in_proj_qkvz          57.3 -> 55.3 us
    (5120, 6144): (128, 64, 80),     # GDN out_proj / attn o_proj 24.6 -> 23.0 us
    (34816, 5120): (64, 128, 68),    # gate_up (target + drafter)  118.6 -> 113.3 us
    (5120, 17408): (64, 128, 80),    # down (target + drafter)      59.1 -> 58.1 us
    (14336, 5120): (128, 128, 56),   # attention qkv               51.5 -> 50.1 us
    (6144, 5120): (128, 128, 48),    # drafter qkv                 24.9 -> 23.3 us
    (5120, 4096): (128, 64, 80),     # drafter o_proj              18.3 -> 16.6 us
    (5120, 25600): (128, 128, 80),   # drafter fc                  84.4 -> 82.9 us
}


def table():
    return dict(_TABLE)
