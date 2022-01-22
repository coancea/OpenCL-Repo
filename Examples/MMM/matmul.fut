-- ==
-- random input { [1024][1024]f32 [1024][1024]f32 } auto output

-- random input { [2048][4096]f32 [4096][2048]f32 } auto output

-- compiled input @ matmul-data/m10x500x64

let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
  reduce (+) 0f32 (map2 (*) xs ys)

let main [n][m][p] (xss: [n][m]f32) (yss: [m][p]f32): [n][p]f32 =
  map (\xs -> map (dotprod xs) (transpose yss)) xss
