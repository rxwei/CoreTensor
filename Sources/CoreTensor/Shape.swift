//
//  Shape.swift
//  CoreTensor
//
//  Copyright 2016-2017 Richard Wei.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

/// Tensor shape
public struct TensorShape: ExpressibleByArrayLiteral {
    public typealias SubSequence = TensorShape

    /// NOTE: Temporarily changed access from fileprivate to public for use
    /// with RankedTensor.
    public var dimensions: [Int]

    /// Initialize with rank, and set the size of each dimension to 1.
    /// - parameter rank: rank of the tensor
    fileprivate init(rank: Int) {
        dimensions = Array(repeating: 1, count: rank)
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init<C: Collection>(_ dimensions: C) where C.Iterator.Element == Int {
        self.dimensions = Array(dimensions)
    }

    /// Initialize with an array literal, representing the sizes of
    /// dimensions. The rank of the tensor is the length of the parameter
    /// list.
    /// - parameter dimensions: sizes of dimensions
    public init(arrayLiteral elements: Int...) {
        self.init(elements)
    }

    /// Rank of the tensor
    public var rank: Int {
        return dimensions.count
    }

    /// Size of the tensor as a contiguously stored array
    public var contiguousSize: Int {
        return simplified().reduce(1, *)
    }
}

// MARK: - Factories
public extension TensorShape {
    static var scalar: TensorShape {
        return []
    }

    static func vector(ofSize size: Int) -> TensorShape {
        return [size]
    }

    static func matrix(rowCount: Int, columnCount: Int) -> TensorShape {
        return [rowCount, columnCount]
    }
}

// MARK: - Predicates
public extension TensorShape {
    var isScalar: Bool {
        return rank == 0
    }

    var isVector: Bool {
        return rank == 1
    }

    var isMatrix: Bool {
        return rank == 2
    }
}

// MARK: - RandomAccessCollection
extension TensorShape : RandomAccessCollection {
    /// Subshape indexing
    public subscript(bounds: Range<Int>) -> TensorShape {
        get {
            return TensorShape(dimensions[bounds])
        }
        set {
            dimensions[bounds] = ArraySlice(newValue.dimensions)
        }
    }

    public var indices: CountableRange<Int> {
        return dimensions.indices
    }

    public func index(after i: Int) -> Int {
        return dimensions.index(after: i)
    }

    public func index(before i: Int) -> Int {
        return dimensions.index(before: i)
    }

    public var startIndex: Int {
        return dimensions.startIndex
    }

    public var endIndex: Int {
        return dimensions.endIndex
    }

    /// Size of i-th dimension
    /// - parameter i: dimension
    public subscript(i: Int) -> Int {
        get {
            return dimensions[i]
        }
        set {
            dimensions[i] = newValue
        }
    }
}

/// Similarity operators
infix operator ~ : ComparisonPrecedence
infix operator !~ : ComparisonPrecedence

// MARK: - Equatable
extension TensorShape : Equatable {
    public static func ==(lhs: TensorShape, rhs: TensorShape) -> Bool {
        return lhs.dimensions == rhs.dimensions
    }
}

// MARK: - Equality and similarity
public extension TensorShape {
    static func ==(lhs: TensorShape?, rhs: TensorShape) -> Bool {
        return lhs.flatMap { $0 == rhs } ?? false
    }

    static func ==(lhs: TensorShape, rhs: TensorShape?) -> Bool {
        return rhs.flatMap { lhs == $0 } ?? false
    }

    func isSimilar(to other: TensorShape) -> Bool {
        return simplified() == other.simplified()
    }

    /// Determine similarity between shapes
    static func ~(lhs: TensorShape, rhs: TensorShape) -> Bool {
        return lhs.isSimilar(to: rhs)
    }
}

/// Determine similarity between shapes
public func ~(lhs: TensorShape?, rhs: TensorShape?) -> Bool {
    return rhs.flatMap { lhs?.isSimilar(to: $0) } ?? false
}

/// Determine non-similarity between shapes
public func !~(lhs: TensorShape?, rhs: TensorShape?) -> Bool {
    return !(lhs ~ rhs)
}

/// Align shape dimensions and add 1-paddings when necesssary
/// - Steps:
///   1. Simplify (drop higher dimensional paddings and suffix starting with 0)
///   2. Add padding to either left-hand side or right-hand side to match the rank
///      of the other side
/// - Example:
///   (1x3x5, 5x1)
///   => (3x5, 5x1) simplified, no padding needed
///   (5x4x5, 1x7x4x1x9)
///   => (5x4x5, 7x4x1x9) simplified
///   => (1x5x4x5, 7x4x1x9) padded
///
/// - Parameters:
///   - lhs: left hand side
///   - rhs: right hand side
///   - body: closure making use of the the new dimensions
/// - Returns: pair of conformed shapes
public func conformedShapes(_ lhs: TensorShape, _ rhs: TensorShape) -> (TensorShape, TensorShape) {
    let lhs = lhs.simplified()
    let rhs = rhs.simplified()
    if lhs.rank == rhs.rank { return (lhs, rhs) }
    if lhs.rank < rhs.rank {
        var newLhs = TensorShape(rank: rhs.rank)
        newLhs[rhs.rank-lhs.rank..<rhs.rank] = lhs
        return (newLhs, rhs)
    } else {
        var newRhs = TensorShape(rank: lhs.rank)
        newRhs[lhs.rank-rhs.rank..<lhs.rank] = rhs
        return (lhs, newRhs)
    }
}

@available(*, deprecated, message: "Use 'conformedShapes(_:_:)' instead")
public func withConformedShapes<Result>(
    _ lhs: TensorShape, _ rhs: TensorShape,
    _ body: (TensorShape, TensorShape) throws -> Result) rethrows -> Result {
    let (newLhs, newRhs) = conformedShapes(lhs, rhs)
    return try body(newLhs, newRhs)
}

/// Tensor multiplication operator
infix operator ⊗ : MultiplicationPrecedence

/// Matrix multiplication operator
infix operator • : MultiplicationPrecedence

// MARK: - Transformations
public extension TensorShape {
    /// Prepending a dimension
    func prepending(_ dimensionSize: Int) -> TensorShape {
        return TensorShape([dimensionSize] + dimensions)
    }

    /// Dropping a specified dimension
    func droppingDimension(_ dimension: Int) -> TensorShape {
        guard indices.contains(dimension) else { return self }
        var newDims = dimensions
        newDims.remove(at: dimension)
        return TensorShape(newDims)
    }

    /// Dropping 1-paddings from higher dimensions, if any
    func droppingHigherPaddings() -> TensorShape {
        // Would use `drop(while:)` in Swift 3.1
        let nonOneIndex = indices.first(where: { self[$0] != 1 }) ?? endIndex
        return suffix(from: nonOneIndex)
    }

    /// Dropping suffix starting with 0, if any
    func droppingEmptySuffix() -> TensorShape {
        // Would use `prefix(while:)` in Swift 3.1
        let firstZeroIndex = indices.first(where: { self[$0] == 0 }) ?? endIndex
        return prefix(upTo: firstZeroIndex)
    }

    /// Returns the simplified shape, i.e. shape after dropping higher 1-paddings and
    /// suffix starting with 0
    func simplified() -> TensorShape {
        return droppingHigherPaddings().droppingEmptySuffix()
    }

    /// Determine if self can be concatenated with other
    func isConcatenable(with other: TensorShape, alongDimension dim: Int = 0) -> Bool {
        return dimensions.prefix(dim) == other.dimensions.prefix(dim)
            && dimensions.suffix(from: dim+1) == other.dimensions.suffix(from: dim+1)
    }

    /// Concatenate two tensor shapes that have every dimension equal except
    /// the specified dimension to concatenate along (the last dimension, by
    /// default)
    ///
    /// - Parameter other: shape to concatenate with
    /// - Returns: concatenated shape, or nil if dimensions don't match
    func concatenating(with other: TensorShape,
                       alongDimension dim: Int = 0) -> TensorShape? {
        guard isConcatenable(with: other, alongDimension: dim) else { return nil }
        var newShape = self
        newShape[dim] = self[dim] + other[dim]
        return newShape
    }

    /// Determine if self can be tensor-multiplied by other
    func isTensorMultiplicable(by other: TensorShape) -> Bool {
        return last == other.first
    }

    /// Returns the result of tensor multiplication of self and other
    /// or `nil` if shapes mismatch
    func tensorMultiplied(by other: TensorShape) -> TensorShape? {
        guard last == other.first else { return nil }
        let newDim = dimensions.dropLast() + other.dimensions.dropFirst()
        return TensorShape(newDim)
    }

    /// Returns the result of tensor multiplication of self and other
    /// or `nil` if shapes mismatch
    static func ⊗ (lhs: TensorShape, rhs: TensorShape) -> TensorShape? {
        return lhs.tensorMultiplied(by: rhs)
    }

    /// Determine if self can be matrix-multiplied by other
    func isMatrixMultiplicable(by other: TensorShape) -> Bool {
        return rank == other.rank
            && rank >= 2
            && prefix(rank-2) == other.prefix(rank-2)
            && self[rank-1] == other[rank-2]
    }

    /// Returns the result of matrix multiplication of self and other
    /// or `nil` if shapes mismatch
    func matrixMultiplied(by other: TensorShape) -> TensorShape? {
        guard isMatrixMultiplicable(by: other) else { return nil }
        var newShape = self
        newShape[rank-1] = other[rank-1]
        return newShape
    }

    /// Returns the result of matrix multiplication of self and other
    /// or `nil` if shapes mismatch
    static func • (lhs: TensorShape, rhs: TensorShape) -> TensorShape? {
        return lhs.matrixMultiplied(by: rhs)
    }

    /// Transpose shape
    var transpose: TensorShape {
        return TensorShape(reversed())
    }

    /// Broadcast degenerate dimensions
    func broadcast(with other: TensorShape) -> TensorShape? {
        /// If one is scalar, return the other
        if isScalar { return other }
        if other.isScalar { return self }
        /// Rank must be equal
        guard rank == other.rank else { return nil }
        var shape: TensorShape = []
        /// For each pair of corresponding dimensions `l` and `r`, it must either
        /// be the case that `l` is equal to `r`, or that either of the two is 1.
        for (l, r) in zip(self, other) {
            if l == 1 || l == r { shape.dimensions.append(r) } else if r == 1 { shape.dimensions.append(l) } else { return nil }
        }
        return shape
    }

    /// Determine if two shapes are compatible, a.k.a. broadcastable
    func isCompatible(with other: TensorShape) -> Bool {
        return broadcast(with: other) != nil
    }
}

// MARK: - Sequence helpers
extension Sequence {
    /// Returns true if all elements satisfy the predicate
    func forAll(_ predicate: (Iterator.Element) -> Bool) -> Bool {
        return reduce(true, { $0 && predicate($1) })
    }
}
