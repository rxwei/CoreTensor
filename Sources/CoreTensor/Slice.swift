//
//  Tensor.swift
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

/// Tensor
public struct TensorSlice<UnitType> : TensorProtocol {
    public typealias Base = Tensor<UnitType>
    public typealias Element = TensorSlice<UnitType>

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
            baseElemShape.dropFirst(indexingDepth - 1)
        }
    }

    var shape: TensorShape {
        return elementShape?.prepending(count) ?? .scalar
    }

    var isScalar: Bool {
        return base.isScalar
    }

    /// The number of units per element
    var unitCountPerElement: Int {
        return elementShape?.contiguousSize ?? 0
    }

    var units: ArraySlice<UnitType> {
        return bounds.flatMap { bounds in
            base.units[base.unitSubrange(from: bounds)]
        } ?? ArraySlice(base.units)
    }

    var unitCount: Int {
        return count * base.unitCountPerElement
    }

    var indices: CountableRange<Int> {
        return bounds ?? 0..<0
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
        precondition(!(isScalar && bounds != nil),
                     "Slices of a scalar cannot have bounds")
        self.base = base
        self.bounds = bounds
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
        where S.Iterator.Element : TensorProtocol
    {
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
        return lhs.base == rhs.base
    }

    public func elementsEqual(_ other: TensorSlice<UnitType>) -> Bool {
        return base.elementsEqual(other.base)
    }

    public func unitsEqual(_ other: TensorSlice<UnitType>) -> Bool {
        return base.unitsEqual(other.base)
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


