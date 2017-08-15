# Project CoreTensor

**CoreTensor** is a sub-project of the DLVM project, providing *DLVM Core* and *NNKit*
with a library for tensor shaping, storage, indexing, slicing, linear algebra shape
transformation and collection behavior.

## Libraries

- *CoreTensor* implements completely dynamically shaped tensors. Scalars, vectors, matrices
and n-D arrays are all represented as `Tensor<T>`, where `T` represents the type of each unit.
Each `Tensor<T>` stores a `TensorShape`, which wraps an array of integers representing the
shape of the tensor.

- *RankedTensor* implements dynamically shaped but *statically ranked* tensors. Instead of
storing `TensorShape` which wraps a dynamically sized array, *RankedTensor* requires a tuple of
integers representing a shape with known rank. Types `T`, `Tensor1D<T>`, `Tensor2D<T>`, `Tensor3D<T>`, and 
`Tensor4D<T>` represent scalars, vectors, matrices, rank-3 tensors, and rank-4 tensors, respectively.

## License

Apache 2.0
