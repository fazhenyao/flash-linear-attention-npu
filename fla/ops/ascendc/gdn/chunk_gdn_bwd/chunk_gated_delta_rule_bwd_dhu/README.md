# ChunkGatedDeltaRuleBwdDhu

Fresh Ascend C implementation of the reverse chunk state recurrence used by
Flash Linear Attention gated delta rule backward.

Inputs use `[B,Hk,T,K]` for `q/k`, `[B,Hv,T,K]` for `w`, `[B,Hv,T,V]`
for `d_o/dv`, `[B,Hv,T]` for `g`, and `[B,Hv,T,K]` for `gk`. The operator
supports GVA when `Hv % Hk == 0`, optional base-2 gates `g/gk`, optional
final-state gradient `dht`, fixed and variable length sequences, and
`chunk_size=64` or `128`.

On A2/A3, the public wrappers normalize the `gk`-only FP32 case to the main
input dtype before launch. The FP32 gate path remains unchanged when `g` is
also present.

The A2/A3 path assigns one reverse scan to each `(sequence,value-head)` task.
For `V=128`, the FP32 recurrent state stays resident in UB across chunks. The
`V=256` fallback keeps the state in a per-core GM workspace and processes it in
UB tiles. Cube/vector intermediates use separate workspace regions and
cross-core ready/free flags.
