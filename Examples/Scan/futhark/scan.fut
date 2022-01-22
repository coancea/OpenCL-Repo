-- ==
-- entry: scanF32
-- random input {  [16777216]i32 } auto output
-- random input {  [100000000]i32 } auto output

entry scanF32 [n] (inp: [n]i32) =
    scan (+) 0 inp


-- futhark bench --backend=cuda --pass-option=--load-cuda=scan-ker.cu -r 3000 scan.fut
