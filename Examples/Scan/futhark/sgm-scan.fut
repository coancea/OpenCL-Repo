-- ==
-- random input { [1][10000000]i32 } auto output
-- random input { [10][1000000]i32 } auto output
-- random input { [100][100000]i32 } auto output
-- random input { [1000][10000]i32 } auto output
-- random input { [10000][1000]i32 } auto output
-- random input { [100000][100]i32 } auto output
-- random input { [1000000][10]i32 } auto output
-- random input { [10000000][1]i32 } auto output

entry main [m][n] (inp: [m][n]i32) =
    map (scan (+) 0i32) inp
