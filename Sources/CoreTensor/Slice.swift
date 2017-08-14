//
//  Slice.swift
//  CoreTensor
//
//  Copyright 2016-2017 DLVM Team.
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

/// TensorSlice
public struct TensorSlice<UnitType> : TensorProtocol {
    public typealias Base = Tensor<UnitType>

    public private(set) var base: Base
    private var baseIndices: [Int]
    private var bounds: CountableRange<Int>?
}

public extension TensorSlice {
    /// Indexing depth of this slice, i.e. the rank difference between the base
    /// and the slice
    private var indexingDepth: Int {
        return baseIndices.count
    }

    /// Shape of element tensors
    var elementShape: TensorShape? {
        return base.elementShape.flatMap { baseElemShape in
            if baseElemShape.rank + 1 == indexingDepth {
                return nil
            }
            return baseElemShape.dropFirst(indexingDepth)
        }
    }

    var isScalar: Bool {
        return indexingDepth == base.shape.count
    }

    /// The number of units per element
    var unitCountPerElement: Int {
        return elementShape?.contiguousSize ?? 0
    }

    var units: ArraySlice<UnitType> {
        let trimmedShape = base.shape.dropFirst()
        var (start, end) = baseIndices.enumerated().reduce((0, base.unitCount), { (acc, next) in
            let stride = trimmedShape.dropFirst(next.offset).reduce(1, *)
            if next.offset == indexingDepth - 1 {
                let temp = acc.0 + next.element * stride
                return (temp, temp + stride)
            }
            return (acc.0 + next.element * stride, acc.1)
        })
        if let bounds = bounds {
            let stride = trimmedShape.dropFirst(indexingDepth).reduce(1, *)
            (start, end) = (start + bounds.startIndex * stride,
                            start + bounds.endIndex * stride)
        }
        let range = start..<end
        return base.units[range]
    }

    var indices: CountableRange<Int> {
        if let bounds = bounds {
            return bounds
        } else if indexingDepth < base.shape.rank {
            return 0..<(base.shape[indexingDepth])
        }
        return 0..<1
    }

    var startIndex: Int {
        return indices.startIndex
    }

    var endIndex: Int {
        return indices.endIndex
    }

    /// Capacity reserved for element tensors
    var capacity: Int {
        return base.capacity
    }
}

public extension TensorSlice {

    init(base: Tensor<UnitType>, bounds: CountableRange<Int>?) {
        if let bounds = bounds {
            precondition(base.indices ~= bounds.startIndex
                         && base.indices ~= bounds.endIndex - 1,
                         "Slice is out of bounds")
        }
        self.base = base
        self.baseIndices = []
        self.bounds = bounds
    }

    init(base: TensorSlice<UnitType>, bounds: CountableRange<Int>?) {
        if let bounds = bounds {
            precondition(base.indices ~= bounds.startIndex
                         && base.indices ~= bounds.endIndex - 1,
                         "Slice is out of bounds")
        }
        self.base = base.base
        self.baseIndices = base.baseIndices
        self.bounds = bounds
    }

    init(base: Tensor<UnitType>, index: Int) {
        precondition(!base.isScalar,
                     "A scalar has no elements")
        precondition(base.indices ~= index,
                     "Element index is out of bounds")
        self.base = base
        self.baseIndices = [index]
        self.bounds = nil
    }

    init(base: TensorSlice<UnitType>, index: Int) {
        precondition(!base.isScalar,
                     "A scalar has no elements")
        precondition(base.indices ~= index,
                     "Element index is out of bounds")
        self.base = base.base
        self.baseIndices = base.baseIndices + [index]
        self.bounds = nil
    }

    init(base: Tensor<UnitType>, indices: [Int]) {
        precondition(!base.isScalar,
                     "A scalar has no elements")
        precondition(zip(base.shape, indices).filter{ !($1 >= 0 && $1 < $0) }.count == 0,
                     "Element indices are out of bounds")
        self.base = base
        self.baseIndices = indices
        self.bounds = nil
    }

    init(base: TensorSlice<UnitType>, indices: [Int]) {
        precondition(!base.isScalar,
                     "A scalar has no elements")
        precondition(zip(base.shape, indices).filter{ !($1 >= 0 && $1 < $0) }.count == 0,
                     "Element indices are out of bounds")
        self.base = base.base
        self.baseIndices = base.baseIndices + indices
        self.bounds = nil
    }

    init(_ base: Tensor<UnitType>) {
        self.init(base: base, bounds: base.indices)
    }

    init(_ base: TensorSlice<UnitType>) {
        self.init(base: base, bounds: base.indices)
    }

    internal init(elementShape: TensorShape?, units: ContiguousArray<UnitType>) {
        let base = Tensor<UnitType>(elementShape: elementShape, units: units)
        self.init(base: base, bounds: base.indices)
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    fileprivate init(shape: TensorShape, units: ContiguousArray<UnitType>) {
        precondition(units.count >= shape.contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.init(elementShape: shape.isScalar ? nil : shape.dropFirst(),
                  units: ContiguousArray(units.prefix(shape.contiguousSize)))
    }

    /// Initialize an empty tensor of scalar elements
    public init() {
        self.init(elementShape: .scalar)
    }

    /// Initialize an empty tensor
    public init(elementShape: TensorShape) {
        self.init(elementShape: elementShape, units: [])
    }

    /// Initialize a scalar tensor
    public init(scalar: UnitType) {
        self.init(elementShape: nil, units: [scalar])
    }

    /// Initialize a tensor from a sequence of tensors of element shape
    public init<S: Sequence>(elementShape: TensorShape, elements: S)
        where S.Element : TensorProtocol, S.Element.UnitType == UnitType {
        let newBase = Tensor<UnitType>(elementShape: elementShape,
                                       elements: elements)
        self.init(base: newBase, bounds: newBase.indices)
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    public init<S: Sequence>(shape: TensorShape, units: S,
                             vacancySupplier supplier: (() -> UnitType)? = nil)
        where S.Iterator.Element == UnitType, S.SubSequence : Sequence,
        S.SubSequence.Iterator.Element == UnitType {
            let contiguousSize = shape.contiguousSize
            var slice = ContiguousArray(units.prefix(contiguousSize))
            /// If elements fewer than required by the shape and supplier is provided
            /// generate new elements using the supplier until vacancy is filled
            if slice.count < contiguousSize, let supplier = supplier {
                slice.reserveCapacity(shape.contiguousSize)
                slice.append(contentsOf: (0..<contiguousSize).map { _ in supplier() })
            }
            self.init(shape: shape, units: slice)
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: TensorShape, repeating repeatedValue: UnitType) {
        let units = ContiguousArray(repeating: repeatedValue, count: shape.contiguousSize)
        self.init(shape: shape, units: units)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: TensorShape, supplier: () -> UnitType) {
        let units = ContiguousArray((0..<shape.contiguousSize).map { _ in supplier() })
        self.init(shape: shape, units: units)
    }

}

/// - TODO: Add conditional expressibility conformance in Swift 4

public extension TensorSlice where UnitType : Equatable {

    public static func ==(lhs: TensorSlice<UnitType>, rhs: TensorSlice<UnitType>) -> Bool {
        return lhs.elementsEqual(rhs)
    }

    public func elementsEqual(_ other: TensorSlice<UnitType>) -> Bool {
        guard shape == other.shape else { return false }
        return shape.elementsEqual(other.shape) && units.elementsEqual(other.units)
    }

    public func unitsEqual(_ other: TensorSlice<UnitType>) -> Bool {
        return units.elementsEqual(other.units)
    }

}

extension TensorSlice where UnitType : Strideable {

    public init(shape: TensorShape, unitsIncreasingFrom lowerBound: UnitType) {
        var unit = lowerBound
        self.init(shape: shape, supplier: {
            defer { unit = unit.advanced(by: 1) }
            return unit
        })
    }

}

extension TensorSlice where UnitType : Strideable, UnitType.Stride : SignedInteger {

    public init(scalarElementsIn bounds: CountableRange<UnitType>) {
        self.init(elementShape: .scalar, units: ContiguousArray(bounds))
    }

    public init(scalarElementsIn bounds: CountableClosedRange<UnitType>) {
        self.init(elementShape: .scalar, units: ContiguousArray(bounds))
    }

}

extension TensorSlice : RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = TensorSlice<UnitType>
    public typealias SubSequence = TensorSlice<UnitType>

    /// Access the element tensor specified by a TensorIndex
    public subscript(index: TensorIndex) -> TensorSlice<UnitType> {
        get {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            return TensorSlice(base: self, bounds: range)
        }
        set {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            precondition(newShape == newValue.shape, "Shape mismatch")
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            base.units.replaceSubrange(range, with: newValue.units)
        }
    }

    /// Access the element tensor specified by a list of dimensional indices
    /// - parameter indices: tensor indices
    /// - note: the count of indices must equal the raw rank of the tensor
    public subscript(indices: Int...) -> TensorSlice<UnitType> {
        get {
            return self[TensorIndex(indices)]
        }
        set {
            self[TensorIndex(indices)] = newValue
        }
    }

    /// Access the element tensor in the current dimension at an index
    public subscript(index: Int) -> Element {
        get {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            return TensorSlice(base: self, index: index)
        }
        set {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst()
            precondition(newShape == newValue.shape, "Shape mismatch")
            let contiguousIndex = unitIndex(fromIndex: index)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            base.units.replaceSubrange(range, with: newValue.units)
        }
    }

    /// Access the sub-tensor specified by a contiguous range of indices
    public subscript(bounds: Range<Int>) -> SubSequence {
        get {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            return TensorSlice(base: self, bounds: CountableRange(bounds))
        }
        set {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            precondition(newValue.shape == elementShape?.prepending(bounds.count),
                         "Shape mismatch")
            base.units[unitSubrange(from: CountableRange(bounds))] =
                newValue.base.units[newValue.unitSubrange(from: newValue.indices)]
        }
    }
}

public extension TensorSlice {

    public func reshaped(as newShape: TensorShape) -> Tensor<UnitType>? {
        guard self.shape.contiguousSize == newShape.contiguousSize else {
            return nil
        }
        return Tensor(shape: newShape, units: units)
    }

}

public extension TensorSlice {

    func withUnsafeBufferPointer<Result>
        (_ body: (UnsafeBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try units.withUnsafeBufferPointer { ptr in
            try body(ptr)
        }
    }

    mutating func withUnsafeMutableBufferPointer<Result>
        (_ body: (inout UnsafeMutableBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        var units = self.units
        return try units.withUnsafeMutableBufferPointer { ptr in
            try body(&ptr)
        }
    }

}

extension TensorSlice : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write(
            isScalar ? String(describing: units.first!)
                : "[\(map {"\($0)"}.joined(separator: ", "))]"
        )
    }
}
