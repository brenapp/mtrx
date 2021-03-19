#![feature(const_fn)]

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// The requirements for a type to be a Matrix Cell. Numeric types fulfill these
/// requirements, and many of them can be derived as needed 
pub trait MatrixCell<T>: Add<Output=T> + Mul<Output=T> + AddAssign + SubAssign + Copy + From<i8> {}
impl<T: Add<Output=T> + Mul<Output=T> + AddAssign + SubAssign + Copy + From<i8>> MatrixCell<T> for T {} 


/// Uses const generics to represent a mathematical matrix 
#[derive(Copy, Clone)]
pub struct Matrix<T: MatrixCell<T>, const R: usize, const C: usize> {
    pub inner: [[T; C]; R]
}

impl<T: MatrixCell<T>, const R: usize, const C: usize> Matrix<T, R, C> {

    /// Creates a new Matrix of the given size
    ///
    /// # Arguments
    ///
    /// * 'inner' - The initial value of the matrix, defines the dimensions of the matrix
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    ///
    /// let matrix = Matrix::new([[1, 2], [3, 4]]); 
    ///
    /// ```
    pub fn new(inner: [[T; C]; R]) -> Self {
        Matrix {
            inner
        }
    }

    /// Returns a 2D array representation of the matrix
    pub fn inner(&self) -> [[T; C]; R] {
        self.inner
    }


    /// Returns true if the matrix is a square matrix
    /// 
    /// # Examples
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let a = Matrix::new([[1, 2], [3, 4]]);
    /// let b = Matrix::new([[1, 2, 3], [3, 4, 5]]);
    /// assert!(a.is_square());
    /// assert!(!b.is_square());
    /// 
    /// ```
    pub const fn is_square(&self) -> bool {
        R == C
    }
    
    /// Multiples the matrix by a scalar value, and returns a new matrix with the scaled values
    /// 
    /// # Arguments
    /// 
    /// * 'scalar' - Value to multiply the matrix by
    ///
    /// Returns a matrix with dimensions R×C (unchanged)
    ///
    /// # Examples
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let matrix = Matrix::new(
    ///     [[1, 1], [2, 2]]
    /// );
    /// 
    /// let result = matrix.multiply_scalar(2);
    /// assert_eq!(result.inner(), [[2, 2], [4, 4]]); 
    /// ```
    pub fn multiply_scalar(&self, scalar: T) -> Matrix<T, R, C> {
        
        // Use the default value
        let mut inner = [[0i8.into(); C]; R];

        for r in 0..R {
            for c in 0..C {
                inner[r][c] = scalar * self.inner[r][c];
            };
        };

        Matrix { inner }

    }

    /// Performs the dot product with the row of this matrix and the column of the given matrix,
    /// used in matrix multiplication
    fn dot_product<const K: usize>(&self, row: usize, matrix: Matrix<T, C, K>, col: usize) -> T {

        // Initialize sum with the first value so that we never have to initialize it with zero, which can be difficult with generic numeric types
        let mut sum: T = self.inner[row][0] * matrix.inner[0][col];

        // Add the remainder of the n-tuple
        for i in 1..C { 
            sum += self.inner[row][i] * matrix.inner[i][col]
        }

        // Return the sum
        sum        
    }



    /// Performs matrix multiplication with the given matrix, returns the resultant matrix
    /// 
    /// # Arguments
    ///
    /// * 'matrix' Matrix of dimensions C×K 
    ///
    /// Returns a matrix with dimensions R×K
    /// 
    /// # Examples
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let matrix_a = Matrix::new(
    ///     [[1, 2, 3], 
    ///     [4, 5, 6]]
    /// );
    ///
    /// let matrix_b = Matrix::new(
    ///     [[7,  8],
    ///      [9,  10], 
    ///      [11, 12]]
    /// );
    ///
    /// let result = matrix_a.multiply_matrix(matrix_b);
    /// assert_eq!(result.inner, [[58, 64], [139, 154]]); 
    ///  
    /// ```
    /// 
    pub fn multiply_matrix<const K: usize>(&self, matrix: Matrix<T, C, K>) -> Matrix<T, R, K> {

        // Initialize a default array (the default values are just placeholders)
        let mut inner = [[0i8.into(); K]; R];

        // Perform the multiplication
        for r in 0..R {
            for c in 0..K {
                inner[r][c] = self.dot_product(r, matrix, c);
            }
        }

        Matrix { inner }
        
    }


    /// Returns the transposed matrix.
    /// 
    /// Matrix transposition is the process of "rotating" the matrix 90 degrees, essentially
    /// swapping rows and columns. For example,
    /// 
    /// 
    /// | 1 2 |
    /// | 3 4 |
    /// | 5 6 |
    /// 
    /// becomes
    /// 
    /// | 1 3 5 |
    /// | 2 4 6 |
    /// 
    /// Returns the transposed Matrix<C, R>
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let matrix = Matrix::new(
    ///     [[1, 2, 3], 
    ///     [4, 5, 6]]
    /// );
    /// 
    /// let transposed = matrix.transpose();
    /// 
    /// assert_eq!(transposed.inner, [[1, 4], [2, 5], [3, 6]])
    /// 
    /// ```
    ///  
    pub fn transpose(&self) -> Matrix<T, C, R> {

        let mut inner = [[0i8.into(); R]; C];

        for r in 0..R {
            for c in 0..C {
                inner[c][r] = self.inner[r][c];
            }
        }


        Matrix { inner }

    }


    /// Adds two matrices of the same size and returns the sum matrix (also the same size).
    /// Additionally you can also use the + operator to add matrices together;
    /// 
    /// # Arguments
    /// 
    /// * 'other' - Same same sized matrix to add
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let matrix_a = Matrix::new([[1, 2], [3, 4]]);
    /// let matrix_b = Matrix::new([[3, 2], [1, 0]]);
    /// 
    /// let sum = matrix_a.add_matrix(matrix_b);
    /// assert_eq!(sum.inner, [[4, 4], [4, 4]]);
    /// 
    /// let sum = matrix_a + matrix_b;
    /// assert_eq!(sum.inner, [[4, 4], [4, 4]]);
    /// 
    /// ```
    pub fn add_matrix(&self, other: Matrix<T, R, C>) -> Matrix<T, R, C> {
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] += other.inner[r][c];
            }
        }

        Matrix { inner }
    }

    /// Adds a single value to all cells in the matrix and returns the sum matrix. Additionally, you
    /// can use the plus operator to add a value to the matrix
    /// 
    /// # Arguments
    /// 
    /// * 'other' - The value T to add to all the cell
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// let matrix_a = Matrix::new([[1, 2], [3, 4]]);
    /// 
    /// let sum = matrix_a.add_value(10);
    /// assert_eq!(sum.inner, [[11, 12], [13, 14]]);
    /// 
    /// let sum = matrix_a + 10;
    /// assert_eq!(sum.inner, [[11, 12], [13, 14]]);
    /// 
    /// ```
    pub fn add_value(&self, other: T) -> Matrix<T, R, C> {
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] += other
            }
        }

        Matrix { inner }
    }

    /// Subtracts two matrices of the same size and returns the difference matrix (also the same size).
    /// Additionally you can also use the - operator to subtract matrices.
    /// 
    /// # Arguments
    /// 
    /// * 'other' - Same same sized matrix to subtract
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let matrix_a = Matrix::new([[1, 2], [3, 4]]);
    /// let matrix_b = Matrix::new([[0, 1], [2, 3]]);
    /// 
    /// let difference = matrix_a.sub_matrix(matrix_b);
    /// assert_eq!(difference.inner, [[1, 1], [1, 1]]);
    /// 
    /// let difference = matrix_a - matrix_b;
    /// assert_eq!(difference.inner, [[1, 1], [1, 1]]);
    /// 
    /// ```
    pub fn sub_matrix(&self, other: Matrix<T, R, C>) -> Matrix<T, R, C> {
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] -= other.inner[r][c];
            }
        }

        Matrix { inner }
    }

    /// Subtracts a single value to all cells in the matrix and returns the difference matrix. Additionally, you
    /// can use the - operator to add a value to the matrix
    /// 
    /// # Arguments
    /// 
    /// * 'other' - The value T to subtract from each cell
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// let matrix_a = Matrix::new([[1, 2], [3, 4]]);
    /// 
    /// let sum = matrix_a.sub_value(1);
    /// assert_eq!(sum.inner, [[0, 1], [2, 3]]);
    /// 
    /// let sum = matrix_a - 1;
    /// assert_eq!(sum.inner, [[0, 1], [2, 3]]);
    /// 
    /// ```
    pub fn sub_value(&self, other: T) -> Matrix<T, R, C> {
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] -= other
            }
        }

        Matrix { inner }
    }


    /// Multiplies a matrix by a mathematical vector (const-sized array) and returns the matrix
    /// vector product.  
    ///
    /// # Arguments
    /// 
    /// * 'other' - Mathematical vector to multiply the matrix by
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let matrix = Matrix::new([
    ///     [1, -1, 2],
    ///     [0, -3, 1]
    /// ]);
    /// 
    /// let vector = [2, 1, 0];
    /// 
    /// let product = matrix.vector_product(vector);
    /// assert_eq!(product, [1, -3]);
    /// 
    /// let product = matrix * vector;
    /// assert_eq!(product, [1, -3])
    /// 
    /// 
    /// ```
    /// 
    pub fn vector_product(&self, other: [T; C]) -> [T; R] {

        let mut values = [0i8.into(); R];

        for r in 0..R {
            for c in 0..C {
                values[r] += self.inner[r][c] * other[c]
            }
        }

        values

    } 

    /// Returns a non-mutable reference to the cell at the specified row and column
    /// 
    /// Note: typical mathematical notation for matrices is for 1-indexing. However, in order to be
    /// consistent, this function is zero-indexed.
    /// 
    /// # Arguments
    /// 
    /// * 'row' - Must be within 0..R
    /// * 'col' - Must be within 0..C
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let matrix = Matrix::new([
    ///     [1, 2], 
    ///     [3, 4]
    /// ]);
    /// 
    /// assert_eq!(matrix.get(0, 0), Some(&1));
    /// assert_eq!(matrix.get(0, 1), Some(&2));
    /// assert_eq!(matrix.get(1, 0), Some(&3));
    /// assert_eq!(matrix.get(1, 1), Some(&4));
    /// 
    /// assert_eq!(matrix.get(2, 2), None);
    /// 
    /// ```
    /// 
    pub const fn get(&self, row: usize, col: usize) -> Option<&T> {

        if row < R && col < C {
            Some(&self.inner[row][col])
        } else {
            None
        }

    }

    /// Returns a mutable reference to the cell at the specified row and column. 
    /// 
    /// Note: typical mathematical notation for matrices is for 1-indexing. However, in order to be
    /// consistent, this function is zero-indexed.
    /// 
    /// # Arguments
    /// 
    /// * 'row' - Must be within 0..R
    /// * 'col' - Must be within 0..C
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let mut matrix = Matrix::new([
    ///     [1, 2], 
    ///     [3, 4]
    /// ]);
    /// 
    /// 
    /// assert_eq!(matrix.get_mut(0, 0), Some(&mut 1));
    /// assert_eq!(matrix.get_mut(0, 1), Some(&mut 2));
    /// assert_eq!(matrix.get_mut(1, 0), Some(&mut 3));
    /// assert_eq!(matrix.get_mut(1, 1), Some(&mut 4));
    /// 
    /// assert_eq!(matrix.get_mut(2, 2), None);
    /// 
    /// ```
    /// 
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {

        if row < R && col < C {
            Some(&mut self.inner[row][col])
        } else {
            None
        }

    }

    /// Sets the value of the cell at the given dimensions
    /// 
    /// Note: typical mathematical notation for matrices is for 1-indexing. However, in order to be
    /// consistent, this function is zero-indexed.
    /// 
    /// # Arguments
    /// 
    /// * 'row' - Must be within 0..R
    /// * 'col' - Must be within 0..C
    /// * 'value' - The value to set at row, col
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let mut matrix = Matrix::new([
    ///     [1, 2], 
    ///     [3, 4]
    /// ]);
    /// 
    /// matrix.set(0, 0, 0);
    /// assert_eq!(matrix.get(0, 0), Some(&0));
    /// 
    /// ```
    /// 
    pub fn set(&mut self, row: usize, col: usize, value: T) -> bool {

        if let Some(cell) = self.get_mut(row, col) {
            *cell = value;
            true
        } else {
            false
        }

    }

}


/// Some matrix operations are only valid for square matrices
impl<T: MatrixCell<T>, const R: usize> Matrix<T, R, R> {

    /// Returns the identity matrix for a RxR matrix. The identity matrix is the matrix with 1 in a
    /// diagonal line down the matrix, and a zero everywhere else. For example, the 3x3 identity
    /// matrix is:
    /// 
    /// 1 0 0
    /// 
    /// 0 1 0
    /// 
    /// 0 0 1
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let identity: Matrix<i8, 2, 2> = Matrix::identity();
    /// assert_eq!(identity.inner, [[1, 0], [0, 1]]);
    /// 
    /// ```
    /// 
    /// Identity matrices cannot be created with non-square const generic sizes:
    /// 
    /// ```compile_fail
    /// use mtrx::Matrix;
    /// 
    /// let identity: Matrix<i8, 3, 2> = Matrix::identity(); // Compiler Error!
    /// 
    /// ```
    pub fn identity() -> Matrix<T, R, R> {

        let mut inner = [[0i8.into(); R]; R];

        for r in 0..R {
            for c in 0..R {
                if r == c {
                    inner[r][c] = 1i8.into();
                }
            }
        };

        Matrix { inner }
    }

    /// Raises a square matrix to a power (essentially, multiplying itself exp times). Raising a
    /// matrix to the zeroth power returns [Matrix::identity]
    /// 
    /// # Arguments
    /// 
    /// * 'exp' - Exponent
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mtrx::Matrix;
    /// 
    /// let matrix = Matrix::new([[1, -3], [2, 5]]);
    /// let result = matrix.pow(2);
    /// 
    /// assert_eq!(result.inner, [[-5, -18], [12, 19]]);
    /// 
    /// let result = matrix.pow(0);
    /// assert_eq!(result.inner, [[1, 0], [0, 1]]);
    /// 
    /// 
    /// ```
    pub fn pow(&self, exp: usize) -> Matrix<T, R, R> {

        // By convention matrix to the power of zero is the identity matrix
        if exp == 0 {
            Matrix::identity()

        // Otherwise, multiply the matrix by itself exp times 
        } else {
            let mut matrix = self.clone();

            for _ in 1..exp {
                matrix = matrix * matrix;
            };
            
            matrix
        }
    }

}

/// Trait Implementations
/// See method descriptions above for more details

impl<T: MatrixCell<T>, const R: usize, const C: usize> Add for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    
    fn add(self, other: Self) -> Self {
        self.add_matrix(other)
    }
}


impl<T: MatrixCell<T>, const R: usize, const C: usize> Add<T> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn add(self, other: T) -> Self {
        self.add_value(other)
    }

}


impl<T: MatrixCell<T>, const R: usize, const C: usize> Sub for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    
    fn sub(self, other: Self) -> Self {
        self.sub_matrix(other)
    }
}

impl<T: MatrixCell<T>, const R: usize, const C: usize> Sub<T> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn sub(self, other: T) -> Self {
        self.sub_value(other)
    }

}


impl<T: MatrixCell<T>, const R: usize, const C: usize, const K: usize> Mul<Matrix<T, C, K>> for Matrix<T, R, C> {
    type Output = Matrix<T, R, K>;

    fn mul(self, other: Matrix<T, C, K>) -> Matrix<T, R, K> {
        self.multiply_matrix(other)
    }
}


impl<T: MatrixCell<T>, const R: usize, const C: usize> Mul<T> for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn mul(self, other: T) -> Matrix<T, R, C> {
        self.multiply_scalar(other)
    }
}

impl<T: MatrixCell<T>, const R: usize, const C: usize> Mul<[T; C]> for Matrix<T, R, C> {
    type Output = [T; R];

    fn mul(self, other: [T; C]) -> [T; R] {
        self.vector_product(other)
    }
}
