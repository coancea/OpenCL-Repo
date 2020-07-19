-- ==
-- compiled input @ matmul-data/m10x500x64

-- notune compiled input @ matmul-data/2pow25_work_2pow0_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow1_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow2_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow3_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow4_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow5_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow6_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow7_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow8_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow9_outer
-- notune compiled input @ matmul-data/2pow25_work_2pow10_outer
--
-- compiled input @ matmul-data/2pow20_work_2pow0_outer
-- compiled input @ matmul-data/2pow20_work_2pow1_outer
-- compiled input @ matmul-data/2pow20_work_2pow2_outer
-- compiled input @ matmul-data/2pow20_work_2pow3_outer
-- compiled input @ matmul-data/2pow20_work_2pow4_outer
-- compiled input @ matmul-data/2pow20_work_2pow5_outer
-- compiled input @ matmul-data/2pow20_work_2pow6_outer
-- compiled input @ matmul-data/2pow20_work_2pow7_outer
-- compiled input @ matmul-data/2pow20_work_2pow8_outer
-- compiled input @ matmul-data/2pow20_work_2pow9_outer
-- compiled input @ matmul-data/2pow20_work_2pow10_outer


let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
  reduce (+) 0f32 (map2 (*) xs ys)

let main (xss: [10][64]f32) (yss: [64][500]f32): [10][500]f32 =
  map (\xs -> map (dotprod xs) (transpose yss)) xss
