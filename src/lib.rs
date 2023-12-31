pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Clone)]
pub struct Matrix {
    pub rows: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        let val = self.rows.get(row)?.get(col)?;
        Some(*val)
    }

    pub fn shape(&self) -> (usize, usize) {
        let rows: usize;
        let cols: usize = self.rows.get(0).expect("Failed to get len").len();
        if cols == 0 {
            rows = 0;
        } else {
            rows = self.rows.len();
        }
        (rows, cols)
    }

    pub fn transpose(&self) -> Matrix {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let (nrows, ncols) = self.shape();
        for cc in 0..ncols {
            let mut row: Vec<f64> = Vec::new();
            for rc in 0..nrows {
                row.push(self.get(rc, cc).expect("Failed to get value"));
            }
            rows.push(row);
        }
        Matrix { rows }
    }
    /// Matrix multiplication
    /// The dot product of two matrices is the sum of the products of the corresponding entries of the two matrices.
    /// # Arguments
    ///  - self: Matrix
    ///  - other: Matrix
    /// - return: Matrix
    /// # Examples
    /// ```
    /// use rustmath::Matrix;
    /// let mat1: Matrix = Matrix {
    ///    rows: vec![
    ///       vec![1.0, 2.0],
    ///      vec![3.0, 4.0],
    ///   ],
    /// };
    /// let mat2: Matrix = Matrix {
    ///   rows: vec![
    ///      vec![5.0, 6.0],
    ///     vec![7.0, 8.0],
    /// ],
    /// };
    /// let expected: Matrix = Matrix {
    ///   rows: vec![
    ///     vec![19.0, 22.0],
    ///    vec![43.0, 50.0],
    /// ],
    /// };
    /// let result = mat1.dot(&mat2);
    /// assert_eq!(result, expected);
    /// ```
    pub fn dot(&self, other: &Matrix) -> Matrix {
        //
        let (nrows, ncols) = self.shape();
        let (orows, ocols) = other.shape();
        assert_eq!(ncols, orows);
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for rc in 0..nrows {
            let mut row: Vec<f64> = Vec::new();
            for cc in 0..ocols {
                let mut sum: f64 = 0.0;
                for i in 0..ncols {
                    sum += self.get(rc, i).expect("Failed to get value") * other.get(i, cc).expect("Failed to get value");
                }
                row.push(sum);
            }
            rows.push(row);
        }
        Matrix { rows }
    }

    pub fn from_vec(vec: Vec<Vec<f64>>) -> Matrix {
        Matrix { rows: vec }
    }

    pub fn diag(&self) -> Vec<f64> {
        let (nrows, _ncols) = self.shape();
        let mut diag: Vec<f64> = Vec::new();
        for i in 0..nrows {
            diag.push(self.get(i, i).expect("Failed to get value"));
        }
        diag
    }

    pub fn fill(n: usize, value: f64) -> Matrix {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for _i in 0..n {
            let mut row: Vec<f64> = Vec::new();
            for _j in 0..n {
                row.push(value);
            }
            rows.push(row);
        }
        Matrix { rows }
    }

    pub fn ones(n: usize) -> Matrix {
        Matrix::fill(n, 1.0)
    }

    pub fn zeros(n: usize) -> Matrix {
        Matrix::fill(n, 0.0)
    }

    pub fn to_diag(vec: Vec<f64>) -> Matrix {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for i in 0..vec.len() {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..vec.len() {
                if i == j {
                    row.push(vec[i]);
                } else {
                    row.push(0.0);
                }
            }
            rows.push(row);
        }
        Matrix { rows }
    }
    
    pub fn identity(n: usize) -> Matrix {
        Matrix::to_diag(vec![1.0; n])
    }

    pub fn inverse(&self) -> Result<Matrix, &'static str> {
        let (nrows, ncols) = self.shape();
        if nrows != ncols {
           return Err("Matrix is not square");
        }
        assert_eq!(nrows, ncols);

        let mut mat = self.clone(); // Clone the matrix to avoid modifying the original
        let mut inv = Matrix::identity(nrows); // Start with the identity matrix

        for i in 0..nrows {
            let pivot = mat.get(i, i).expect("Failed to get value");
            
            if pivot.abs() < 1e-10 {
                return Err("Matrix is singular");
            }

            // Scale the pivot row
            for j in 0..ncols {
                mat.rows[i][j] /= pivot;
                inv.rows[i][j] /= pivot;
            }

            // Eliminate other rows
            for j in 0..nrows {
                if i != j {
                    let factor = mat.get(j, i).expect("Failed to get value");
                    for k in 0..ncols {
                        mat.rows[j][k] -= factor * mat.get(i, k).expect("Failed to get value");
                        inv.rows[j][k] -= factor * inv.get(i, k).expect("Failed to get value");
                    }
                }
            }
        }

        Ok(inv)
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
    #[test]
    fn test_matrix_get() {
        let mat = Matrix {
            rows: vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
        };
        assert_eq!(mat.get(1, 2), Some(6.0));
    }

    #[test]
    fn test_matrix_shape() {
        let mat = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![4.0, 5.0],
                vec![7.0, 8.0],
            ],
        };
        assert_eq!(mat.shape(), (3, 2));

        let mat = Matrix {
            rows: vec![vec![]],
        };
        let (rows, cols) = mat.shape();
        println!("{} {}", rows, cols);
        assert_eq!(mat.shape(), (0, 0));
    }

    #[test]
    fn test_matrix_transpose() {
        let mat0: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![4.0, 5.0],
                vec![7.0, 8.0],
            ],
        };
        let mat1: Matrix = Matrix {
            rows: vec![
                vec![1.0, 4.0, 7.0],
                vec![2.0, 5.0, 8.0],
            ],
        };
        let mat2: Matrix = Matrix {
            rows: vec![
                vec![1.0, 4.0],
                vec![2.0, 5.0],
                vec![7.0, 8.0],
            ],
        };
        let mat_t: Matrix = mat0.transpose();
        assert_eq!(mat_t, mat1);
        assert_ne!(mat_t, mat2);
    }

    #[test]
    fn test_matrix_dot() {
        let mat1: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
            ],
        };
        let mat2: Matrix = Matrix {
            rows: vec![
                vec![5.0, 6.0],
                vec![7.0, 8.0],
            ],
        };
        let expected: Matrix = Matrix {
            rows: vec![
                vec![19.0, 22.0],
                vec![43.0, 50.0],
            ],
        };
        let result = mat1.dot(&mat2);
        assert_eq!(result, expected);
    }
    // Test function for diag
    #[test]
    fn test_matrix_diag() {
        let mat: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
            ],
        };
        let expected: Vec<f64> = vec![1.0, 4.0];
        let result = mat.diag();
        assert_eq!(result, expected);
    }
    #[test]
    fn test_math_1() {
        let mat: Matrix = Matrix {
            rows: vec![
                vec![4.0, 0.0],
                vec![3.0, -5.0],
            ],
        };
        let mat_t: Matrix = mat.transpose();
        let mat_t_mat: Matrix = mat_t.dot(&mat);
        let expected: Matrix = Matrix {
            rows: vec![
                vec![25.0, -15.0],
                vec![-15.0, 25.0],
            ],
        }; 
        assert_eq!(mat_t_mat, expected);
    }
    #[test]
    fn test_ones() {
        let mat: Matrix = Matrix::ones(3);
        let expected: Matrix = Matrix {
            rows: vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ],
        };
        assert_eq!(mat, expected);
    }
    #[test]
    fn test_to_diag() {
        let mat: Matrix = Matrix::to_diag(vec![1.0, 2.0, 3.0]);
        let expected: Matrix = Matrix {
            rows: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0],
                vec![0.0, 0.0, 3.0],
            ],
        };
        assert_eq!(mat, expected);
    }
    #[test]
    fn test_inverse() {
        let mat_foo: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
            ],
        };
        let expected: Matrix = Matrix {
            rows: vec![
                vec![-2.0, 1.0],
                vec![1.5, -0.5],
            ],
        };
        assert_eq!(mat_foo.inverse().expect(""), expected);

        // test singular matrix
        let mat_bar: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![2.0, 4.0],
            ],
        };
        assert!(mat_bar.inverse().is_err());
    }
    #[test]
    fn test_fill() {
        let mat: Matrix = Matrix::fill(3, 1.0);
        let expected: Matrix = Matrix {
            rows: vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ],
        };
        assert_eq!(mat, expected);
    }
}