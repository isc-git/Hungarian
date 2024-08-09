use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hungarian::hungarian;

pub fn criterion_benchmark(c: &mut Criterion) {
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

    c.bench_function("hungarian", |b| {
        b.iter(|| black_box(hungarian(black_box(&mut costs.clone()))))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
