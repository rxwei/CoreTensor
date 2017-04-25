//
//  Index.swift
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
    func contiguousIndex(for index: TensorIndex) -> Int {
        return index.contiguousIndex(in: self)
    }

    /// Returns the shape after indexing
    subscript(index: TensorIndex) -> TensorShape? {
        guard index.count < rank else {
            return nil
        }
        return dropFirst(index.count)
    }

}
