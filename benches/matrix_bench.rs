use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustmath::Matrix;

fn matrix_operations_benchmark(c: &mut Criterion) {
    // Test matrices of different sizes
    let sizes = [2, 10, 50, 100];
    
    for &size in sizes.iter() {
        let mut group = c.benchmark_group(format!("Matrix Size {0}x{0}", size));
        
        // Create test matrices
        let mat_a = Matrix::fill(size, 2.0);
        let mat_b = Matrix::fill(size, 3.0);
        
        // Benchmark matrix multiplication
        group.bench_function("multiplication", |b| {
            b.iter(|| black_box(&mat_a).dot(black_box(&mat_b)))
        });
        
        // Benchmark matrix transpose
        group.bench_function("transpose", |b| {
            b.iter(|| black_box(&mat_a).transpose())
        });
        
        // Benchmark matrix inverse
        group.bench_function("inverse", |b| {
            b.iter(|| black_box(&mat_a).inverse())
        });
        
        // Benchmark SVD
        group.bench_function("svd", |b| {
            b.iter(|| black_box(&mat_a).svd())
        });
        
        // Benchmark LU decomposition
        group.bench_function("lu_decomposition", |b| {
            b.iter(|| black_box(&mat_a).lu_decomposition())
        });
        
        group.finish();
    }
}

criterion_group!(benches, matrix_operations_benchmark);
criterion_main!(benches);