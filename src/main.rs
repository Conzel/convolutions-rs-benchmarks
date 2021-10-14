use std::time::Instant;

use convolutions_rs::convolutions::*;
use convolutions_rs::transposed_convolutions::*;
use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn main() {
    let mut small_duration = 0u128;
    let test_cycles_small = 1_000_000;
    // small input images
    for _ in 0..test_cycles_small {
        let x = Array::random((1, 10, 10), Uniform::new(-1., 1.));
        let k = Array::random((3, 1, 3, 3), Uniform::new(-1., 1.));

        let now = Instant::now();
        let _ = conv2d(&k, &x, convolutions_rs::Padding::Same, 1);
        small_duration += now.elapsed().as_nanos();
    }
    println!(
        "Time for small arrays, {} iterations: {} milliseconds",
        test_cycles_small,
        small_duration / 1_000_000
    );

    let mut medium_duration = 0u128;
    let test_cycles_medium = 10_000;
    // medium input images
    for _ in 0..test_cycles_medium {
        let x = Array::random((1, 100, 100), Uniform::new(-1., 1.));
        let k = Array::random((3, 1, 3, 3), Uniform::new(-1., 1.));

        let now = Instant::now();
        let _ = conv2d(&k, &x, convolutions_rs::Padding::Same, 1);
        medium_duration += now.elapsed().as_nanos();
    }
    println!(
        "Time for medium arrays, {} iterations: {} milliseconds",
        test_cycles_medium,
        medium_duration / 1_000_000
    );

    let mut large_duration = 0u128;
    let test_cycles_large = 10;
    // large input images
    for _ in 0..test_cycles_large {
        let x = Array::random((1, 1000, 10000), Uniform::new(-1., 1.));
        let k = Array::random((3, 1, 3, 3), Uniform::new(-1., 1.));

        let now = Instant::now();
        let _ = conv2d(&k, &x, convolutions_rs::Padding::Same, 1);
        large_duration += now.elapsed().as_nanos();
    }
    println!(
        "Time for large arrays, {} iterations: {} milliseconds",
        test_cycles_large,
        large_duration / 1_000_000
    );
}
