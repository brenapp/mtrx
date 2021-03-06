use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::default::Default;

/// The requirements for a type to be a Matrix Cell. Numeric types fulfill these
/// requirements, and many of them can be derived as needed 
pub trait MatrixCell<T>: Add<Output=T> + Mul<Output=T> + AddAssign + SubAssign + Default + Copy {}
impl<T: Add<Output=T> + Mul<Output=T> + AddAssign + SubAssign + Default + Copy> MatrixCell<T> for T {} 

/// Uses const generics to represent a mathematical matrix 
#[derive(Copy, Clone)]
pub struct Matrix<T: MatrixCell<T>, const R: usize, const C: usize> {
    inner: [[T; C]; R]
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
    /// // Generates a new 2x2 matrix 
    /// let matrix = Matrix::new([[1, 2], [3, 4]]); 
    ///
    /// ```
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
    /// Returns a matrix with dimensions R×C (unchanged)
    ///
    /// # Examples
    /// ```
    /// let matrix = Matrix::new(
    ///     [[1, 1], [2, 2]]
    /// );
    /// 
    /// let result = matrix.multiply_scalar(2);
    /// assert_eq!(result.inner, [[2, 2], [4, 4]]); 
    /// ```
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
    pub fn multiply<const K: usize>(&self, matrix: Matrix<T, C, K>) -> Matrix<T, R, K> {

        // Initialize a default array (the default values are just placeholders)
        let mut inner = [[T::default(); K]; R];

        // Perform the multiplication
        for r in 0..R {
            for c in 0..K {
                inner[r][c] = self.dot_product(r, matrix, c);
            }
        }

        Matrix { inner }
        
    }


    /// Returns the transposed matrix.
    pub fn transpose(&self) -> Matrix<T, C, R> {

        let mut inner = [[T::default(); R]; C];

        for r in 0..R {
            for c in 0..C {
                inner[r][c] = self.inner[c][r];
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
        self.multiply(other)
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

    #[test]
    pub fn transpose() {

    }

}