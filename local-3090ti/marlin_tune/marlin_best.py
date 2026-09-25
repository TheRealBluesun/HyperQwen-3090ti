"""qwen27-dev: M<=8 Marlin CTA counts for Qwen3.8-27B W4A16 + DFlash2 on an RTX 3090 @ 350 W
(measured with bench sweeps of sms 30..82; only entries >1% better than the default 82 are kept)."""
_TABLE = {
    (16384, 5120): 64,   # GDN in_proj_qkvz      57.3 -> 55.5 us
    (34816, 5120): 68,   # gate_up (target, drafter) 117.7 -> 114.0 us
    (5120, 17408): 80,   # down                  58.9 -> 58.3 us
    (14336, 5120): 56,   # attention qkv         51.3 -> 50.1 us
    (6144, 5120): 49,    # drafter qkv           24.7 -> 23.2 us
    (5120, 4096): 40,    # drafter o_proj        18.1 -> 17.9 us
}


def table():
    return dict(_TABLE)
