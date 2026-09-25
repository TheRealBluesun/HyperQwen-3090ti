"""Prepare a standalone Marlin build (bf16 x uint4b8 only) exposing thread_k / thread_n / sms."""
import os, re
G = os.environ.get("MT_GEN", "gen"); O = os.environ.get("MT_OUT", "mt"); os.makedirs(O, exist_ok=True)
for f in ("kernel.h", "marlin.cuh", "marlin_dtypes.cuh", "dequant.h", "marlin_mma.h", "marlin_template.h"):
    open(os.path.join(O, f), "w").write(open(os.path.join(G, f)).read())
keep = lambda s: ("a_type == vllm::kBFloat16" in s and "vllm::kU4B8" in s and "thread_m_blocks == 1" in s and "group_blocks == 8" in s) if "==" in s else \
                 ("kBFloat16.id(), vllm::kU4B8.id(), vllm::kBFloat16.id(), vllm::kBFloat16.id()" in s)
# selector: pairs of lines (condition, assignment)
lines = open(os.path.join(G, "kernel_selector.h")).read().splitlines()
out = [l for l in lines if l.startswith("//")]
first = True
i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith("if (") or l.startswith("else if ("):
        cond, asg = l, lines[i + 1]
        if keep(cond):
            cond = re.sub(r"^else if \(", "if (", cond) if first else re.sub(r"^if \(", "else if (", cond)
            out += [cond, asg]; first = False
        i += 2
    else:
        i += 1
open(os.path.join(O, "kernel_selector.h"), "w").write("\n".join(out) + "\n")
nsel = sum(1 for l in out if "kernel =" in l)
# instantiations: the template lines whose template args match (thread_m_blocks=1, group_blocks=8)
src = open(os.path.join(G, "sm80_kernel_bfloat16_u4b8_bfloat16.cu")).read().splitlines()
kept = []
for l in src:
    if l.startswith("template __global__ void Marlin<"):
        args = [a.strip() for a in l[l.index("<") + 1:l.index(">(")].split(",")]
        # a,b,c,s,threads,thread_m_blocks,thread_n_blocks,thread_k_blocks,m_block_size_8,stages,group_blocks,is_zp_float
        if args[5] == "1" and args[10] == "8":
            kept.append(l)
    else:
        kept.append(l)
open(os.path.join(O, "kernels.cu"), "w").write("\n".join(kept) + "\n")
ninst = sum(1 for l in kept if l.startswith("template __global__"))
# host code: marlin.cu minus stable-ABI parts, cut before the real wrapper
m = open(os.path.join(G, "marlin.cu")).read()
for inc in ("#include <torch/csrc/stable/accelerator.h>", "#include <torch/csrc/stable/library.h>", "#include <torch/csrc/stable/ops.h>",
            "#include <torch/csrc/stable/tensor.h>", "#include <torch/headeronly/core/ScalarType.h>", '#include "libtorch_stable/torch_utils.h"'):
    m = m.replace(inc + "\n", "")
second = [x.start() for x in re.finditer(r"\ntorch::stable::Tensor marlin_gemm\(", m)][1]
m = m[:second] + "\n#endif\n"
# the arch<750 stub references torch::stable in a never-compiled branch; drop it for the host pass too
m = re.sub(r"torch::stable::Tensor marlin_gemm\([\s\S]*?return torch::stable::empty\(\{1, 1\}\);\n\}\n", "", m, count=1)
open(os.path.join(O, "host.cu"), "w").write(m)
print(f"selector entries {nsel}, instantiations {ninst}")
