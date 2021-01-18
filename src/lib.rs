use std::ops::{Add, Mul, AddAssign};
use std::default::Default;

#[derive(Copy, Clone)]
pub struct Matrix<T: Add<Output=T> + Mul<Output=T> + AddAssign + Default + Copy, const R: usize, const C: usize> {
    inner: [[T; C]; R]
}

impl<T: Add<Output=T> + Mul<Output=T> + AddAssign + Default + Copy, const R: usize, const C: usize> Matrix<T, R, C> {

    /// Creates a new Matrix of the given size
    pub fn new(inner: [[T; C]; R]) -> Self {
        Matrix {
            inner
        }
    }
    
    /// Multiples the matrix by a scalar value, and returns a new matrix with the scaled values
    /// 
    /// # Arguments
    /// 
    /// * 'scalar' - Value to multiply the matrix by
    ///
    /// Returns a matrix with dimensions R×K
    /// 
    pub fn multiply_scalar(&self, scalar: T) -> Matrix<T, R, C> {
        
        // Use the default value
        let mut inner = [[T::default(); C]; R];

        for r in 0..R {
            for c in 0..C {
                inner[r][c] = scalar * self.inner[r][c];
            };
        };

        Matrix { inner }

    }

    /// Performs the dot product with the row of this matrix and the column of the given matrix,
    /// used in matrix multiplication
    /// 
    /// # Arguments
    /// 
    pub fn dot_product<const K: usize>(&self, row: usize, matrix: Matrix<T, C, K>, col: usize) -> T {

        // Initalize sum with the first value so that we never have to initalize it with zero, which can be difficult with generic numeric types
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
    pub fn multiply<const K: usize>(&self, matrix: Matrix<T, C, K>) -> Matrix<T, R, K> {

        // Initalize a default array (the default values are just placeholders)
        let mut inner = [[T::default(); K]; R];

        // Perform the multiplication
        for r in 0..R {
            for c in 0..K {
                inner[r][c] = self.dot_product(r, matrix, c);
            }
        }

        Matrix { inner }
        
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
    }

    #[test]
    pub fn multiply() {

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

        let result = matrix_a.multiply(matrix_b);
        assert_eq!(result.inner, 
            [[58, 64], 
             [139, 154]]
        ); 
    }

}