# convolutions-rs-benchmarks

We provide speed benchmarks for the convolutions-rs crate (https://github.com/Conzel/convolutions-rs).
We have tested the following three situations:

- Small: size (10,10), iterations 10'000'000
- Medium: size (100,100), iterations 10'000
- Large: size (1000,1000), iterations 10

## Results

### convolutions-rs
- Small: 18.58s
- Medium: 21.18s
- Large: 22.74s

### Pytorch
- Small: 16.06s 
- Medium: 0.74s
- Large: 0.29s
