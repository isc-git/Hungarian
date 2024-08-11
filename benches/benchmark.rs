use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use hungarian::{hungarian, Allocations};

pub fn standard_benchmark(c: &mut Criterion) {
    #[rustfmt::skip]
    let costs = nalgebra::Matrix5::from_row_slice(
        &[
            20., 15., 18., 20., 25.,
            18., 20., 12., 14., 15.,
            21., 23., 25., 27., 25.,
            17., 18., 21., 23., 20.,
            18., 18., 16., 19., 20.,
        ]
    );

    let mut assignments = Allocations::default();

    c.bench_function("hungarian", |b| {
        b.iter(|| hungarian(black_box(&costs), black_box(&mut assignments)))
    });
}

pub fn random_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_of_size");
    for size in (1..8).map(|i| 2usize.pow(i)) {
        let mut assignments = Allocations::default();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched_ref(
                || nalgebra::DMatrix::<f64>::new_random(size, size),
                |costs| hungarian(black_box(costs), black_box(&mut assignments)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

pub fn random_benchmarks_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_i32_of_size");
    for size in (1..7).map(|i| 2usize.pow(i)) {
        let mut assignments = Allocations::default();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched_ref(
                || nalgebra::DMatrix::<i32>::new_random(size, size),
                |costs| hungarian(black_box(costs), black_box(&mut assignments)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    standard_benchmark,
    random_benchmarks,
    random_benchmarks_i32
);
criterion_main!(benches);
