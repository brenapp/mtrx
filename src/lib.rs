#![feature(const_fn)]

/// Represents a mathematical matrix, with dimensions known at compile time due to const generics

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// The requirements for a type to be a Matrix Cell. Numeric types fulfill these
/// requirements, and many of them can be derived as needed 
pub trait MatrixCell<T>: Add<Output=T> + Mul<Output=T> + AddAssign + SubAssign + Copy + From<usize> {}
impl<T: Add<Output=T> + Mul<Output=T> + AddAssign + SubAssign + Copy + From<usize>> MatrixCell<T> for T {} 


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
        let mut inner = [[0.into(); C]; R];

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
        let mut inner = [[0.into(); K]; R];

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
    pub fn transpose(&self) -> Matrix<T, C, R> {

        let mut inner = [[0.into(); R]; C];

        for r in 0..R {
            for c in 0..C {
                inner[c][r] = self.inner[r][c];
            }
        }


        Matrix { inner }

    }


    /// Adds two matrices of the same size and returns the sum matrix (also the same size)
    pub fn add_matrix(&self, other: Matrix<T, R, C>) -> Matrix<T, R, C> {
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] += other.inner[r][c];
            }
        }

        Matrix { inner }
    }

    /// Adds a single value to all cells in the matrix and returns the sum matrix
    pub fn add_value(&self, other: T) -> Matrix<T, R, C> {
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] += other
            }
        }

        Matrix { inner }
    }

    /// Adds two matrices of the same size and returns the resultant matrix (also the same size)
    pub fn sub_matrix(&self, other: Matrix<T, R, C>) -> Matrix<T, R, C> {
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] -= other.inner[r][c];
            }
        }

        Matrix { inner }
    }

    /// Adds a single value to all cells in the matrix and returns the sum matrix
    pub fn sub_value(&self, other: T) -> Matrix<T, R, C> {
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] -= other
            }
        }

        Matrix { inner }
    }


    /// Multiplies a matrix by a vector (const-sized array) and returns the matrix vector product.  
    pub fn vector_product(&self, other: [T; C]) -> [T; R] {

        let mut values = [0.into(); R];

        for r in 0..R {
            for c in 0..C {
                values[r] += other[c]
            }
        }

        values

    } 

    /// Returns a non-mutable reference to the cell at the specified row and column
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {

        if row < R && col < C {
            Some(&self.inner[row][col])
        } else {
            None
        }

    }

    /// Returns a mutable reference to the cell at the specified row and column. 
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {

        if row < R && col < C {
            Some(&mut self.inner[row][col])
        } else {
            None
        }

    }

    /// Sets the value of the reference 
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

    pub fn identity() -> Matrix<T, R, R> {

        let mut inner = [[0.into(); R]; R];

        for r in 0..R {
            for c in 0..R {
                if r == c {
                    inner[r][c] = 1.into();
                }
            }
        };

        Matrix { inner }
    }

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
        
        let mut inner = self.inner.clone();

        for r in 0..R {
            for c in 0..C {
                inner[r][c] -= other.inner[r][c];
            }
        }

        Matrix { inner }

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


#[cfg(test)]
mod tests {
    use crate::Matrix;
    
    #[test]
    pub fn multiply_scalar() {

        let matrix = Matrix::new(
            [[1, 1], [2, 2]]
        );


        let result = matrix.multiply_scalar(2);
        assert_eq!(result.inner, [[2, 2], [4, 4]]); 


        let result = matrix * 2;
        assert_eq!(result.inner, [[2, 2], [4, 4]]); 

    }

    #[test]
    pub fn multiply_matrix() {

        let matrix_a = Matrix::new(
            [[1, 2, 3], 
             [4, 5, 6]]
        );

        let matrix_b = Matrix::new(
            [[7,  8],
             [9,  10], 
             [11, 12],
            ]
        );

        let result = matrix_a.multiply_matrix(matrix_b);
        assert_eq!(result.inner, 
            [[58, 64], 
             [139, 154]]
        ); 
    }

    #[test]
    pub fn transpose() {

        let matrix = Matrix::new(
            [
                [1, 2, 3], 
                [4, 5, 6]
            ]
        );


        let result = matrix.transpose();
        assert_eq!(result.inner, [
            [1, 4], [2, 5], [3, 6]])

    }   

}