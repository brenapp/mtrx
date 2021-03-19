# Mtrx

Matrix operations using Rust's new const generics feature. Matrix sizes are determined at compile
time, allowing better type checking. 

Supported Operations
- Addition
- Subtraction
- Scalar Multiplication
- Matrix Multiplication
- Matrix Vector Product
- Transposition
- Matrix Powers

> **Note: currently, mtrx requires Nightly to work, as it makes use of the `#![feature(const_fn)]`**

```Rust
let matrix_a = Matrix::new(
    [[1, 2, 3], 
    [4, 5, 6]]
);

let matrix_b = Matrix::new(
    [[7,  8],
     [9,  10], 
     [11, 12]]
);

let result: Matrix<i32, 2, 2> = matrix_a.multiply(matrix_b);
assert_eq!(result.inner, 
    [[58, 64], 
     [139, 154]]
); 


let result = matrix_a * matrix_b;
assert_eq!(result.inner, 
    [[58, 64], 
     [139, 154]]
); 

```

