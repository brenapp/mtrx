#![feature(min_const_generics)]

pub struct Matrix<T, const R: usize, const C: usize> {
    inner: [[T; C]; R]
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {

    pub fn new(inner: [[T; C]; R]) -> Self {
        Matrix {
            inner
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::Matrix;


    pub fn basics() {

        let matrix = Matrix::new(
            [[1, 1], 
             [2, 2]]
        );


    }


}