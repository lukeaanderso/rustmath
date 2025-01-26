use std::ops::{Mul, Add};

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Clone)]
pub struct Matrix {
    pub rows: Vec<Vec<f64>>,
}

pub struct SVDResult {
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub vt: Vec<f64>
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

    /// Extracts the diagonal elements of the matrix.
    ///
    /// This function returns a vector containing the elements along the main diagonal
    /// of the matrix (from top-left to bottom-right).
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the Matrix instance.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` containing the diagonal elements of the matrix.
    ///
    /// # Panics
    ///
    /// This function will panic if it fails to retrieve any diagonal element.
    pub fn diag(&self) -> Vec<f64> {
        let (nrows, _ncols) = self.shape();
        let mut diag: Vec<f64> = Vec::new();
        for i in 0..nrows {
            diag.push(self.get(i, i).expect("Failed to get value"));
        }
        diag
    }

    // Matrix addition implementation
    fn add_matrix(&self, other: &Matrix) -> Matrix {
        let (nrows, ncols) = self.shape();
        let (orows, ocols) = other.shape();
        assert_eq!(nrows, orows);
        assert_eq!(ncols, ocols);
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for rc in 0..nrows {
            let mut row: Vec<f64> = Vec::new();
            for cc in 0..ncols {
                row.push(self.get(rc, cc).expect("Failed to get value") + other.get(rc, cc).expect("Failed to get value"));
            }
            rows.push(row);
        }
        Matrix { rows }
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        // Subtraction is the same as adding the negative
        self.add_matrix(&(other * -1.0))
    }

    pub fn similar(&self, other: &Matrix, tol: f64) -> bool {
        let (nrows, ncols) = self.shape();
        let (orows, ocols) = other.shape();
        if nrows != orows || ncols != ocols {
            return false;
        }
        let diff = self.sub(other);
        for rc in 0..nrows {
            for cc in 0..ncols {
                if diff.get(rc, cc).unwrap().abs() > tol {
                    return false;
                }
            }
        }
        return true
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

    pub fn foo() -> i32 {
        let bar:i32 = 3;
        bar
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
    pub fn determinant(&self) -> f64 {
        let (nrows, ncols) = self.shape();
        assert_eq!(nrows, ncols);
        let mut mat: Matrix = self.clone();
        let mut det: f64 = 1.0;
        for i in 0..nrows {
            let pivot = mat.get(i, i).expect("Failed to get value");
            if pivot.abs() < 1e-10 {
                return 0.0;
            }
            det *= pivot;
            for j in 0..ncols {
                mat.rows[i][j] /= pivot;
            }
            for j in 0..nrows {
                if i != j {
                    let factor = mat.get(j, i).expect("Failed to get value");
                    for k in 0..ncols {
                        mat.rows[j][k] -= factor * mat.get(i, k).expect("Failed to get value");
                    }
                }
            }
        }
        det
    }
    pub fn lu_decomposition(&self) -> (Matrix, Matrix) {
        let (nrows, ncols) = self.shape();
        assert_eq!(nrows, ncols);

        let mut l = Matrix::identity(nrows);
        let mut u = self.clone();

        for i in 0..nrows {
            for j in i+1..nrows {
                let factor = u.rows[j][i] / u.rows[i][i];
                l.rows[j][i] = factor;
                for k in i..nrows {
                    u.rows[j][k] -= factor * u.rows[i][k];
                }
            }
        }

        (l, u)
    }
    /// Unpack rows to long vector
    pub fn unpack(&self) -> Vec<f64> {
        let mut vec: Vec<f64> = Vec::new();
        for row in &self.rows {
            for val in row {
                vec.push(*val);
            }
        }
        vec
    }

    pub fn reshape(&self, nrows: usize, ncols: usize) -> Matrix {
        let mut vec: Vec<f64> = self.unpack();
        vec.reverse();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for _ in 0..nrows {
            let mut row: Vec<f64> = Vec::new();
            for _ in 0..ncols {
                row.push(vec.pop().expect("Failed to pop value"));
            }
            rows.push(row);
        }
        Matrix { rows }
    }

    /// Singular Value Decomposition (SVD) of a square matrix.
    /// 
    /// Computes the SVD decomposition A = U·Σ·Vᵀ using LAPACK's dgesdd routine.
    /// The input matrix A is decomposed into:
    /// - U: Left singular vectors (orthogonal matrix)
    /// - Σ: Diagonal matrix of singular values in descending order
    /// - Vᵀ: Transposed right singular vectors (orthogonal matrix)
    /// 
    /// # Implementation Details
    /// - Internally converts matrices between row-major (Rust) and column-major (LAPACK) formats
    /// - Uses LAPACK's divide-and-conquer SVD implementation for better performance
    /// 
    /// # Arguments
    /// * `self` - A square matrix to decompose
    /// 
    /// # Returns
    /// * `Ok(SVDResult)` containing U, singular values, and Vᵀ
    /// * `Err` if the matrix is not square or if the computation fails
    /// 
    /// # Example
    /// ```
    /// use rustmath::Matrix;
    /// let mat = Matrix::from_vec(vec![
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0],
    /// ]);
    /// let svd = mat.svd().unwrap();
    /// // Original matrix can be reconstructed as U·Σ·Vᵀ
    /// ```
    pub fn svd(&self) -> Result<SVDResult, &'static str> {
        let (nrows, ncols) = self.shape();
        assert_eq!(nrows, ncols);
        
        // Convert input matrix to column-major format for LAPACK
        let mut mat = vec![0.0; nrows * ncols];
        for i in 0..nrows {
            for j in 0..ncols {
                mat[i + j * nrows] = self.get(i, j).unwrap();
            }
        }
        
        let mut s: Vec<f64> = vec![0.0; nrows];
        let mut u: Vec<f64> = vec![0.0; nrows * nrows];
        let mut vt: Vec<f64> = vec![0.0; nrows * nrows];
        
        let mut work = vec![0.0; 1];
        let mut iwork: Vec<i32> = vec![0; 8 * nrows];
        let mut info: i32 = 0;
        
        unsafe {
            // Query optimal workspace size
            lapack::dgesdd(
                b'A' as u8,
                nrows as i32,
                ncols as i32,
                &mut mat,
                nrows as i32,
                &mut s,
                &mut u,
                nrows as i32,
                &mut vt,
                nrows as i32,
                &mut work,
                -1,
                &mut iwork,
                &mut info,
            );
            
            if info != 0 {
                return Err("Workspace query failed");
            }
            
            // Allocate optimal workspace and compute SVD
            let lwork = work[0] as usize;
            work = vec![0.0; lwork];
            
            lapack::dgesdd(
                b'A' as u8,
                nrows as i32,
                ncols as i32,
                &mut mat,
                nrows as i32,
                &mut s,
                &mut u,
                nrows as i32,
                &mut vt,
                nrows as i32,
                &mut work,
                lwork as i32,
                &mut iwork,
                &mut info,
            );
            
            match info {
                0 => {
                    // Convert U and VT from column-major back to row-major format
                    let mut u_row_major = vec![0.0; nrows * nrows];
                    let mut vt_row_major = vec![0.0; nrows * nrows];
                    
                    for i in 0..nrows {
                        for j in 0..nrows {
                            u_row_major[i * nrows + j] = u[i + j * nrows];
                            vt_row_major[i * nrows + j] = vt[i + j * nrows];
                        }
                    }
                    
                    Ok(SVDResult { 
                        u: u_row_major, 
                        s, 
                        vt: vt_row_major 
                    })
                },
                i if i < 0 => Err("Invalid argument to dgesdd"),
                _ => Err("SVD computation did not converge"),
            }
        }
    }

    pub fn covariance(&self) -> Matrix {
        let (_nrows, ncols) = self.shape();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for i in 0..ncols {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..ncols {
                row.push(self.get(i, j).expect("Failed to get value"));
            }
            rows.push(row);
        }
        Matrix { rows }
    }

    pub fn mul_scalar(&self, scalar: f64) -> Matrix {
        let (nrows, ncols) = self.shape();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for rc in 0..nrows {
            let mut row: Vec<f64> = Vec::new();
            for cc in 0..ncols {
                row.push(self.get(rc, cc).expect("Failed to get value") * scalar);
            }
            rows.push(row);
        }
        Matrix { rows }
    }

    pub fn add_scalar(&self, scalar: f64) -> Matrix {
        let (nrows, ncols) = self.shape();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for rc in 0..nrows {
            let mut row: Vec<f64> = Vec::new();
            for cc in 0..ncols {
                row.push(self.get(rc, cc).expect("Failed to get value") + scalar);
            }
            rows.push(row);
        }
        Matrix { rows }
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Matrix {
        self.mul_scalar(scalar)
    }
}

impl Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Matrix {
        &self * scalar
    }
}

impl Add<f64> for &Matrix {
    type Output = Matrix;

    fn add(self, scalar: f64) -> Matrix {
        self.add_scalar(scalar)
    }
}

impl Add<f64> for Matrix {
    type Output = Matrix;

    fn add(self, scalar: f64) -> Matrix {
        &self + scalar
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        self.add_matrix(other)
    }
}

impl Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        &self + &other
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

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
    #[test]
    fn test_determinant() {
        let mat: Matrix = Matrix {
            rows: vec![
                vec![6.0, 1.0, 1.0],
                vec![4.0, -2.0, 5.0],
                vec![2.0, 8.0, 7.0],

            ],
        };
        assert_eq!(mat.determinant(), -306.0);
    }
    #[test]
    fn test_lu_decomposition() {
        let mat: Matrix = Matrix {
            rows: vec![
                vec![4.0, 3.0],
                vec![6.0, 3.0],
            ],
        };
        let (l, u) = mat.lu_decomposition();
        let expected_l: Matrix = Matrix {
            rows: vec![
                vec![1.0, 0.0],
                vec![1.5, 1.0],
            ],
        };
        let expected_u: Matrix = Matrix {
            rows: vec![
                vec![4.0, 3.0],
                vec![0.0, -1.5],
            ],
        };
        assert_eq!(l, expected_l);
        assert_eq!(u, expected_u);
    }
    #[test]
    fn test_svd() {
        let mat: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
            ],
        };
        let svd_result = mat.svd().expect("SVD failed");
        
        // Reshape the vectors into matrices
        let u_mat = Matrix::from_vec(vec![
            vec![svd_result.u[0], svd_result.u[1]],
            vec![svd_result.u[2], svd_result.u[3]]
        ]);
        
        let s_mat = Matrix::to_diag(svd_result.s.clone());
        
        let vt_mat = Matrix::from_vec(vec![
            vec![svd_result.vt[0], svd_result.vt[1]],
            vec![svd_result.vt[2], svd_result.vt[3]]
        ]);
        
        // Reconstruct A = U·Σ·Vᵀ
        let us = u_mat.dot(&s_mat);
        let reconstructed = us.dot(&vt_mat);
        
        // Check if the reconstruction matches the original
        assert!(mat.similar(&reconstructed, 1e-6));
        
        // Verify singular values
        assert!((svd_result.s[0] - 5.4649).abs() < 1e-4);
        assert!((svd_result.s[1] - 0.3659).abs() < 1e-4);
    }

    #[test]
    fn test_sub() {
        let mat_a: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
            ],
        };
        let mat_b: Matrix = Matrix {
            rows: vec![
                vec![5.0, 5.0],
                vec![0.0, 4.0],
            ],
        };
        let mat_result: Matrix = Matrix {
            rows: vec![
                vec![-4.0, -3.0],
                vec![3.0, 0.0],
            ],
        };
        assert_eq!(mat_a.sub(&mat_b), mat_result);
    }

    #[test]
    fn test_similar() {
        let mat_a: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
            ],
        };
        let mat_b: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
            ],
        };

        let mat_c: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.1],
            ],
        };

        assert!(mat_a.similar(&mat_b, 1e-10));
        assert!(!mat_a.similar(&mat_c, 1e-10));
        assert!(mat_a.similar(&mat_c, 0.2));
    }

    #[test]
    fn test_reshape() {
        let mat_a: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
            ],
        };
        let mat_b = mat_a.reshape(3, 2);
        let mat_c: Matrix = Matrix {
            rows: vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 6.0],
            ],
        };
        assert_eq!(mat_b, mat_c);
    }

    #[test]
    fn test_svd_simple() {
        // This matrix has known singular values
        let mat_a: Matrix = Matrix {
            rows: vec![
                vec![2.0, 2.0],
                vec![1.0, 1.0],
            ],
        };

        // 1. Get SVD
        let svd_result = mat_a.svd().expect("SVD failed");
        
        // Reshape the vectors into matrices
        let u_mat = Matrix::from_vec(vec![
            vec![svd_result.u[0], svd_result.u[1]],
            vec![svd_result.u[2], svd_result.u[3]]
        ]);
        
        let s_mat = Matrix::to_diag(svd_result.s.clone());
        
        let vt_mat = Matrix::from_vec(vec![
            vec![svd_result.vt[0], svd_result.vt[1]],
            vec![svd_result.vt[2], svd_result.vt[3]]
        ]);
        
        // Print raw values
        println!("Original matrix: {:?}", mat_a);
        println!("U matrix: {:?}", u_mat);
        println!("S matrix: {:?}", s_mat);
        println!("VT matrix: {:?}", vt_mat);
        
        // Check properties that must be true:
        // 1. U and V should be orthogonal (U·Uᵀ = I and V·Vᵀ = I)
        let u_ut = u_mat.dot(&u_mat.transpose());
        let vt_v = vt_mat.dot(&vt_mat.transpose());
        println!("U·Uᵀ (should be identity): {:?}", u_ut);
        println!("VT·VTᵀ (should be identity): {:?}", vt_v);
        
        // 2. The reconstruction
        let us = u_mat.dot(&s_mat);
        let reconstructed = us.dot(&vt_mat);
        println!("Reconstructed matrix: {:?}", reconstructed);
        
        // 3. Singular values
        println!("Singular values: {:?}", svd_result.s);
    }

    #[test]
    fn test_scalar_operations() {
        let mat = Matrix::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);

        // Test scalar multiplication using operator
        let mul_result = &mat * 2.0;
        let expected_mul = Matrix::from_vec(vec![
            vec![2.0, 4.0],
            vec![6.0, 8.0],
        ]);
        assert_eq!(mul_result, expected_mul);

        // Test scalar addition using operator
        let add_result = &mat + 1.0;
        let expected_add = Matrix::from_vec(vec![
            vec![2.0, 3.0],
            vec![4.0, 5.0],
        ]);
        assert_eq!(add_result, expected_add);

        // Test with owned Matrix
        let owned_mul = mat.clone() * 2.0;
        assert_eq!(owned_mul, expected_mul);
        
        let owned_add = mat + 1.0;
        assert_eq!(owned_add, expected_add);
    }

    #[test]
    fn test_matrix_operations() {
        let mat1 = Matrix::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let mat2 = Matrix::from_vec(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);

        // Test matrix addition using operator
        let add_result = &mat1 + &mat2;
        let expected_add = Matrix::from_vec(vec![
            vec![6.0, 8.0],
            vec![10.0, 12.0],
        ]);
        assert_eq!(add_result, expected_add);

        // Test matrix subtraction using refactored method
        let sub_result = mat1.sub(&mat2);
        let expected_sub = Matrix::from_vec(vec![
            vec![-4.0, -4.0],
            vec![-4.0, -4.0],
        ]);
        assert_eq!(sub_result, expected_sub);

        // Test with owned matrices
        let owned_add = mat1.clone() + mat2.clone();
        assert_eq!(owned_add, expected_add);
    }
}
