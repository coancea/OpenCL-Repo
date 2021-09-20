-- ==
-- entry: scanF32
-- random input {  [16777216]f32 } auto output
-- random input {  [33300000]f32 } auto output

entry scanF32 [n] (inp: [n]f32) =
    scan (+) 0.0 inp


-- futhark bench --backend=cuda --pass-option=--load-cuda=scan-ker.cu -r 3000 scan.fut
