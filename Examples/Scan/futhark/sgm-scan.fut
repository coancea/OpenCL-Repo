-- ==
-- random input { [1][100000000]i32 } auto output
-- random input { [10][10000000]i32 } auto output
-- random input { [100][1000000]i32 } auto output
-- random input { [1000][100000]i32 } auto output
-- random input { [10000][10000]i32 } auto output
-- random input { [100000][1000]i32 } auto output
-- random input { [1000000][100]i32 } auto output
-- random input { [10000000][10]i32 } auto output
-- random input { [100000000][1]i32 } auto output

entry main [m][n] (inp: [m][n]f32) =
    map (scan (+) 0.0f32) inp
