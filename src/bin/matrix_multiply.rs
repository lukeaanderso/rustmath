use rand::Rng;
use rustmath::Matrix;
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    // Define matrix dimensions
    let rows_a = 3;
    let cols_a = 4;
    let cols_b = 2;

    // Generate random matrices
    let mut rng = rand::thread_rng();
    
    // Create matrix_a
    let matrix_a = Matrix::from_vec((0..rows_a)
        .map(|_| (0..cols_a)
            .map(|_| rng.gen::<f64>())
            .collect())
        .collect());
    
    // Create matrix_b
    let matrix_b = Matrix::from_vec((0..cols_a)
        .map(|_| (0..cols_b)
            .map(|_| rng.gen::<f64>())
            .collect())
        .collect());

    // Multiply matrices
    let result = matrix_a.dot(&matrix_b);

    // Create/open file for writing
    let mut file = File::create("matrix_result.csv")?;

    // Write the result matrix to CSV
    for row in &result.rows {
        writeln!(file, "{}", row.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(","))?;
    }

    println!("Result has been saved to matrix_result.csv");
    Ok(())
} 