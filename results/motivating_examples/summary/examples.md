# Motivating examples (5G-NIDD, HTTPFlood, seed 0)

| Example | Flow | Attack | p(clean) | p(attacked) | S(x) | S(x') | J top-5 | Qwen clean -> attacked | phi-4 clean -> attacked |
|---|---|---|---|---|---|---|---|---|---|
| harm | s305325 | A1_displacement | 1.000 | 0.991 | sTtl, SrcWin | sTtl, DstWin, TcpRtt, AckDat | 0.75 | SrcWin -> sMeanPktSz | SrcWin -> sMeanPktSz |
| reliance_shift | s316302 | A1_displacement | 1.000 | 0.983 | SrcWin | sTtl, SrcWin, SynAck, AckDat | 0.89 | SrcWin -> AckDat | SrcWin -> AckDat |
| scaffolding | s129551 | A3_scaffolding | 1.000 | 1.000 | sTtl, AckDat | dTtl, TcpRtt | 0.00 | AckDat -> AckDat | AckDat -> AckDat |
