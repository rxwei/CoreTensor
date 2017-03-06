//
//  Shape.swift
//  DLVM
//
//  Created by Richard Wei on 11/13/16.
//
//

/// Tensor shape
public struct TensorShape : ExpressibleByArrayLiteral {

    public typealias SubSequence = TensorShape

    fileprivate var dimensions: [Int]

    /// Initialize with rank, and set the size of each dimension to 1.
    /// - parameter rank: rank of the tensor
    fileprivate init(rank: Int) {
        dimensions = Array(repeating: 1, count: rank)
    }

    /// Initialize with sizes of dimensions. The rank of the tensor
    /// is the length of the parameter list.
    /// - parameter dimensions: sizes of dimensions
    public init<C: Collection>(_ dimensions: C) where C.Iterator.Element == Int {
        precondition(!dimensions.contains(0), "Dimension element is zero")
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

extension TensorShape {

    public static let scalar: TensorShape = []

    public static func vector(ofSize size: Int) -> TensorShape {
        return [size]
    }

    public static func matrix(rowCount: Int, columnCount: Int) -> TensorShape {
        return [rowCount, columnCount]
    }

    public var isScalar: Bool {
        return rank == 0
    }

    public var isVector: Bool {
        return rank == 1
    }

    public var isMatrix: Bool {
        return rank == 2
    }

}

extension TensorShape : RandomAccessCollection {

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

infix operator ~ : ComparisonPrecedence
infix operator !~ : ComparisonPrecedence

extension TensorShape : Equatable {

    public static func ==(lhs: TensorShape, rhs: TensorShape) -> Bool {
        return lhs.dimensions == rhs.dimensions
    }

    public static func ==(lhs: TensorShape?, rhs: TensorShape) -> Bool {
        return lhs.flatMap { $0 == rhs } ?? false
    }

    public static func ==(lhs: TensorShape, rhs: TensorShape?) -> Bool {
        return rhs.flatMap { lhs == $0 } ?? false
    }

    public func isSimilar(to other: TensorShape) -> Bool {
        return simplified() == other.simplified()
    }

    public static func ~(lhs: TensorShape, rhs: TensorShape) -> Bool {
        return lhs.isSimilar(to: rhs)
    }

    public static func ~(lhs: TensorShape?, rhs: TensorShape) -> Bool {
        return lhs.flatMap { $0 ~ rhs } ?? false
    }

    public static func ~(lhs: TensorShape, rhs: TensorShape?) -> Bool {
        return rhs.flatMap { lhs ~ $0 } ?? false
    }

    public static func !~(lhs: TensorShape, rhs: TensorShape) -> Bool {
        return !lhs.isSimilar(to: rhs)
    }

    public static func !~(lhs: TensorShape?, rhs: TensorShape) -> Bool {
        return !(lhs.flatMap { $0 ~ rhs } ?? false)
    }

    public static func !~(lhs: TensorShape, rhs: TensorShape?) -> Bool {
        return !(rhs.flatMap { lhs ~ $0 } ?? false)
    }
    
}

infix operator ⊗ : MultiplicationPrecedence

public func withConformedShapes<Result>(
    _ lhs: TensorShape, _ rhs: TensorShape,
    _ body: (TensorShape, TensorShape) throws -> Result) rethrows -> Result {
    let lhs = lhs.simplified()
    let rhs = rhs.simplified()
    if lhs.rank == rhs.rank { return try body(lhs, rhs) }
    if lhs.rank < rhs.rank {
        var newLhs = TensorShape(rank: rhs.rank)
        newLhs[rhs.rank-lhs.rank..<rhs.rank] = lhs
        return try body(newLhs, rhs)
    } else {
        var newRhs = TensorShape(rank: lhs.rank)
        newRhs[lhs.rank-rhs.rank..<lhs.rank] = rhs
        return try body(lhs, newRhs)
    }
}

public extension TensorShape {

    public func prepending(_ dimensionSize: Int) -> TensorShape {
        return TensorShape([dimensionSize] + dimensions)
    }

    public func droppingDimension(_ dimension: Int) -> TensorShape {
        guard indices.contains(dimension) else { return self }
        var newDims = dimensions
        newDims.remove(at: dimension)
        return TensorShape(newDims)
    }

    public func droppingHigherPaddings() -> TensorShape {
        // Would use `drop(while:)` in Swift 3.1
        let nonOneIndex = indices.first(where: { self[$0] != 1 }) ?? endIndex
        return suffix(from: nonOneIndex)
    }

    public func droppingZeroDimensions() -> TensorShape {
        // Would use `prefix(while:)` in Swift 3.1
        let firstZeroIndex = indices.first(where: { self[$0] == 0 }) ?? endIndex
        return prefix(upTo: firstZeroIndex)
    }

    public func simplified() -> TensorShape {
        return droppingHigherPaddings().droppingZeroDimensions()
    }

    /// Concatenate two tensor shapes that have every dimension equal except
    /// the specified dimension to concatenate along (the last dimension, by
    /// default)
    ///
    /// - Parameter other: shape to concatenate with
    /// - Returns: concatenated shape, or nil if dimensions don't match
    public func concatenating(with other: TensorShape,
                              alongDimension dim: Int = 0) -> TensorShape? {
        /// We do not use conformed shapes here, because two shapes like [1x8] & [1x8] can
        /// be concatenated
        guard self.dimensions.prefix(dim) == other.dimensions.prefix(dim),
            self.dimensions.suffix(from: dim+1) == other.dimensions.suffix(from: dim+1) else {
            return nil // Dimension mismatch
        }
        var newShape = self
        newShape[dim] = self[dim] + other[dim]
        return newShape
    }

    public func canConcatenate(with other: TensorShape) -> Bool {
        return concatenating(with: other) != nil
    }

    public func multiplied(with other: TensorShape) -> TensorShape? {
        return withConformedShapes(self, other) { lhs, rhs in
            guard lhs.last == rhs.first else { return nil }
            let newDim = lhs.dimensions.dropLast() + rhs.dimensions.dropFirst()
            return TensorShape(newDim)
        }
    }

    public static func ⊗ (lhs: TensorShape, rhs: TensorShape) -> TensorShape? {
        return lhs.multiplied(with: rhs)
    }

    public func canMultiply(with other: TensorShape) -> Bool {
        return multiplied(with: other) != nil
    }

    public func matrixMultiplied(with other: TensorShape) -> TensorShape? {
        return withConformedShapes(self, other) { lhs, rhs in
            /// Match inner dimensions for matrix multiplication
            guard lhs.dropFirst().first == rhs.first else { return nil }
            /// Multiply inner dimensions
            var newShape = lhs
            newShape[1] = rhs[1]
            return newShape
        }
    }

    public func canMatrixMultiply(with other: TensorShape) -> Bool {
        return matrixMultiplied(with: other) != nil
    }

    public var transpose: TensorShape {
        return TensorShape(reversed())
    }

    public func broadcasted(to other: TensorShape) -> TensorShape? {
        /// Only scalar broadcasting for now
        return withConformedShapes(self, other) { lhs, rhs in
            lhs ~ .scalar || lhs == rhs ? other : nil
        }
    }

    public func mutuallyBroadcasted(with other: TensorShape) -> TensorShape? {
        return broadcasted(to: other) ?? other.broadcasted(to: self)
    }

    public func canBroadcast(to other: TensorShape) -> Bool {
        return broadcasted(to: other) != nil
    }

    public func canMutuallyBroadcast(with other: TensorShape) -> Bool {
        return mutuallyBroadcasted(with: other) != nil
    }

}

/// Tensor index
public struct TensorIndex : ExpressibleByArrayLiteral {
    var elements: [Int]
    
    public init(arrayLiteral elements: Int...) {
        self.elements = elements
    }

    public init(_ indexElements: Int...) {
        self.elements = indexElements
    }

    public init<S: Sequence>(_ indexElements: S) where S.Iterator.Element == Int {
        self.elements = Array(indexElements)
    }

    public init(repeating repeatedValue: Int, count: Int) {
        self.elements = Array(repeating: repeatedValue, count: count)
    }

    /// Compute the contiguous storage index from high-dimensional tensor indices
    /// - parameter indices: tensor indices
    /// - returns: index in contiguous storage
    /// - note: the count of indices must equal the rank of the tensor
    public func contiguousIndex(in shape: TensorShape) -> Int {
        /// Row-major order addressing
        let trimmedShape = shape.prefix(count)
        return elements.enumerated().reduce(0, { acc, next -> Int in
            let stride = trimmedShape.isEmpty ? 0 : trimmedShape.dropFirst(next.offset+1).reduce(1, *)
            return acc + next.element * stride
        })
    }

}

// MARK: - Equatable, Comparable
extension TensorIndex : Comparable {
    
    public static func ==(lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        return lhs.elements == rhs.elements
    }

    public static func <(lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        for (x, y) in zip(lhs.elements, rhs.elements) {
            /// Less-than at a higher dimension => true
            if x < y { return true }
            /// Greater-than at a higher dimension => false
            if x > y { return false }
            /// Otherwise, at the same higher dimension => continue
        }
        return false
    }
    
}

// MARK: - RandomAccessCollection
extension TensorIndex : RandomAccessCollection {

    public var count: Int {
        return elements.count
    }

    public var dimension: Int {
        return count - 1
    }

    public subscript(bounds: Range<Int>) -> TensorIndex {
        get {
            return TensorIndex(elements[bounds])
        }
        set {
            elements[bounds] = ArraySlice(newValue.elements)
        }
    }

    public var indices: CountableRange<Int> {
        return elements.indices
    }

    public func index(after i: Int) -> Int {
        return elements.index(after: i)
    }

    public func index(before i: Int) -> Int {
        return elements.index(before: i)
    }

    public var startIndex: Int {
        return elements.startIndex
    }

    public var endIndex: Int {
        return elements.endIndex
    }

    /// Size of i-th dimension
    /// - parameter i: dimension
    public subscript(i: Int) -> Int {
        get {
            return elements[i]
        }
        set {
            elements[i] = newValue
        }
    }
    
}

// MARK: - Strideable
extension TensorIndex : Strideable {

    public typealias Stride = Int

    /// Returns a `Self` `x` such that `self.distance(to: x)` approximates `n`.
    ///
    /// If `Stride` conforms to `Integer`, then `self.distance(to: x) == n`.
    ///
    /// - Complexity: O(1).
    public func advanced(by n: Int) -> TensorIndex {
        guard !isEmpty else { return self }
        var newIndex = self
        newIndex[newIndex.endIndex-1] += n
        return newIndex
    }

    /// Returns a stride `x` such that `self.advanced(by: x)` approximates
    /// `other`.
    ///
    /// If `Stride` conforms to `Integer`, then `self.advanced(by: x) == other`.
    ///
    /// - Complexity: O(1).
    public func distance(to other: TensorIndex) -> Int {
        precondition(count == other.count, "Indices are not in the same dimension")
        guard let otherLast = other.last, let selfLast = last else { return 0 }
        return otherLast - selfLast
    }

}

public extension TensorShape {

    /// Returns the row-major order index for the specified tensor index
    /// - parameter index: tensor index
    public func contiguousIndex(for index: TensorIndex) -> Int {
        return index.contiguousIndex(in: self)
    }

    public subscript(index: TensorIndex) -> TensorShape? {
        guard index.count < rank else {
            return nil
        }
        return dropFirst(index.count)
    }
    
}
