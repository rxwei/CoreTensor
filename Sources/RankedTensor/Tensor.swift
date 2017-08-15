//
//  Tensor.swift
//  RankedTensor
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

import struct CoreTensor.TensorShape
import struct CoreTensor.TensorSlice
import struct CoreTensor.Tensor

/// Tensor with static rank
public struct RankedTensor<R: StaticRank> {
    public typealias DataType = R.DataType
    public typealias Shape = R.Shape
    public typealias ElementTensor = R.ElementTensor

    /// Tensor storage
    internal var storage: Tensor<DataType>

    /// Tensor rank
    public var rank: UInt {
        return R.rank
    }

    /// Tensor shape
    public var shape: Shape

    /// The number of items (atom) in the tensor.
    public var unitCount: Int {
        return units.count
    }

    /// The number of items (atoms) per element (sub-tensor).
    /// - Note: This holds the same value as the computed property
    /// `dynamicShape.contiguousSize`, which is extremely frequently used.
    /// We cache it in this variable whenever a new shape is set
    public private(set) var unitCountPerElement: Int

    /// Contiguous storage
    public var units: ContiguousArray<DataType> {
        return storage.units
    }

    /// Capacity reserved for sub-tensor
    public var capacity: Int {
        return units.capacity / unitCountPerElement
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    internal init(shape: Shape, units: ContiguousArray<DataType>) {
        // TODO: Reduce number of calls to `shapeToArray`
        let shapeArray = RankedTensor.shapeToArray(shape)
        let contiguousSize: Int = shapeArray.reduce(1, *)
        precondition(units.count >= contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.shape = shape
        self.storage = Tensor<DataType>(shape: TensorShape(shapeArray), units: units)
        self.unitCountPerElement = shapeArray.count == 0 || shapeArray[0] == 0 ?
            contiguousSize : contiguousSize / shapeArray[0]
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: Shape, repeating repeatedValue: DataType) {
        let contiguousSize: Int = RankedTensor.shapeToArray(shape).reduce(1, *)
        let units = ContiguousArray(repeating: repeatedValue, count: contiguousSize)
        self.init(shape: shape, units: units)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: Shape, supplier: () -> DataType) {
        let contiguousSize: Int = RankedTensor.shapeToArray(shape).reduce(1, *)
        let units = ContiguousArray((0..<contiguousSize).map { _ in supplier() })
        self.init(shape: shape, units: units)
    }
}

public extension RankedTensor {
    internal static func shapeToArray(_ shape: Shape) -> [Int] {
        var shape = shape
        return withUnsafePointer(to: &shape) { ptr in
            ptr.withMemoryRebound(to: UInt.self, capacity: 1) { ptr in
                let buf = UnsafeBufferPointer(start: ptr, count: Int(R.rank))
                return buf.lazy.map {Int($0)}
            }
        }
    }

    var dynamicShape: TensorShape {
        return storage.shape
    }
}

public extension RankedTensor where R.Shape == Shape2D {
    fileprivate func getElementShape() -> Shape1D {
        return arrayToShape1D(Array(dynamicShape.dimensions.dropFirst()))
    }
}

/// - TODO: Add conditional expressibility conformance in Swift 4

public extension RankedTensor where R.DataType : Equatable {
    public static func ==<A>(lhs: RankedTensor<R>, rhs: RankedTensor<A>) -> Bool
        where A.DataType == DataType {
            return lhs.elementsEqual(rhs)
    }

    public func elementsEqual<A>(_ other: RankedTensor<A>) -> Bool
        where A.DataType == DataType {
            guard dynamicShape == other.dynamicShape else { return false }
            return units.elementsEqual(other.units)
    }

    public func unitsEqual<A>(_ other: RankedTensor<A>) -> Bool
        where A.DataType == DataType {
            return units.elementsEqual(other.units)
    }
}

public extension RankedTensor {
    public func isSimilar<A>(to other: RankedTensor<A>) -> Bool
        where A.DataType == DataType {
            return dynamicShape.isSimilar(to: other.dynamicShape)
    }

    public func isIsomorphic<A>(to other: RankedTensor<A>) -> Bool
        where A.DataType == DataType {
            return dynamicShape == other.dynamicShape
    }
}

extension RankedTensor where R.DataType : Strideable {
    public init(shape: Shape, unitsIncreasingFrom lowerBound: DataType) {
        var unit = lowerBound
        self.init(shape: shape, supplier: {
            defer { unit = unit.advanced(by: 1) }
            return unit
        })
    }
}

extension RankedTensor where R.DataType : Strideable, R.DataType.Stride : SignedInteger, R.Shape == (UInt) {
    public init(scalarElementsIn bounds: CountableRange<DataType>) {
        self.init(shape: (UInt(bounds.count)), units: ContiguousArray(bounds))
    }

    public init(scalarElementsIn bounds: CountableClosedRange<DataType>) {
        self.init(shape: (UInt(bounds.count)), units: ContiguousArray(bounds))
    }
}

public extension RankedTensor {
    func makeUnitIterator() -> IndexingIterator<ContiguousArray<DataType>> {
        return units.makeIterator()
    }

    mutating func updateUnit(at index: Int, to newValue: DataType) {
        storage.updateUnit(at: index, to: newValue)
    }
}

public extension RankedTensor where R.DataType : Numeric {
    mutating func incrementUnit(at index: Int, by newValue: DataType) {
        storage.incrementUnit(at: index, by: newValue)
    }

    mutating func decrementUnit(at index: Int, by newValue: DataType) {
        storage.decrementUnit(at: index, by: newValue)
    }

    mutating func multiplyUnit(at index: Int, by newValue: DataType) {
        storage.multiplyUnit(at: index, by: newValue)
    }
}

public extension RankedTensor where R.DataType : BinaryInteger {
    mutating func divideUnit(at index: Int, by newValue: DataType) {
        storage.divideUnit(at: index, by: newValue)
    }
}

public extension RankedTensor where R.DataType : FloatingPoint {
    mutating func divideUnit(at index: Int, by newValue: DataType) {
        storage.divideUnit(at: index, by: newValue)
    }
}

public extension RankedTensor {
    func unitIndex(fromIndex index: Int) -> Int {
        return unitCountPerElement * index
    }

    func unitSubrange(from tensorSubrange: Range<Int>) -> Range<Int> {
        return unitIndex(fromIndex: tensorSubrange.lowerBound)
            ..< unitIndex(fromIndex: tensorSubrange.upperBound)
    }
}

extension RankedTensor : RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = ElementTensor
    public typealias SubSequence = RankedTensorSlice<R>

    /// Get indices corresponding to the units of the element tensor at an index
    fileprivate func getElementTensorRange(index: Int) -> CountableRange<Int> {
        let elementTensorShape = dynamicShape.dropFirst()
        let contiguousIndex = unitIndex(fromIndex: index)
        return contiguousIndex..<contiguousIndex+elementTensorShape.contiguousSize
    }

    /// Get units in the element tensor at an index
    fileprivate func getElementTensorUnits(index: Int) -> ContiguousArray<DataType> {
        let range = getElementTensorRange(index: index)
        return ContiguousArray(units[range])
    }

    /// Access the scalar element or element tensor at an index
    public subscript(index: Int) -> Element {
        get {
            switch (Element.self) {
            case is DataType.Type:
                return units[index] as! Element
            case is TensorSlice1D<DataType>.Type:
                return TensorSlice1D<DataType>(base: self, index: index) as! Element
            case is TensorSlice2D<DataType>.Type:
                return TensorSlice2D<DataType>(base: self, index: index) as! Element
            case is TensorSlice3D<DataType>.Type:
                return TensorSlice3D<DataType>(base: self, index: index) as! Element
            default:
                fatalError("Invalid element tensor type")
            }
        }
        set {
            switch (newValue) {
            case let scalar as DataType:
                storage[index] = TensorSlice(scalar: scalar)
            case let tensor as TensorSlice1D<DataType>:
                storage[index] = TensorSlice(tensor.base)
            case let tensor as TensorSlice2D<DataType>:
                storage[index] = TensorSlice(tensor.base)
            case let tensor as TensorSlice3D<DataType>:
                storage[index] = TensorSlice(tensor.base)
            default:
                fatalError("Invalid element tensor type")
            }
        }
    }

    /// Access the sub-tensor specified by a contiguous range of indices
    public subscript(bounds: Range<Int>) -> SubSequence {
        get {
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            return SubSequence(base: self, bounds: CountableRange(bounds))
        }
        set {
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            precondition(newValue.dynamicShape == dynamicShape.dropFirst().prepending(bounds.count),
                         "Shape mismatch")
            storage[bounds] = newValue.base
        }
    }

    /// Returns the number of elements
    public var count: Int {
        return units.count / unitCountPerElement
    }

    /// Returns a sequence of indices for elements
    public var indices: CountableRange<Int> {
        return 0..<endIndex
    }

    public var startIndex: Int {
        return 0
    }

    public var endIndex: Int {
        return count
    }

    /// Returns the index after the specified one in the current dimension
    public func index(after i: Int) -> Int {
        return i + 1
    }

    /// Returns the index before the specified one in the current dimension
    public func index(before i: Int) -> Int {
        return i - 1
    }
}

public typealias Tensor1D<T> = RankedTensor<R1<T>>
public typealias Tensor2D<T> = RankedTensor<R2<T>>
public typealias Tensor3D<T> = RankedTensor<R3<T>>
public typealias Tensor4D<T> = RankedTensor<R4<T>>

public typealias Vector<T> = Tensor1D<T>
public typealias Matrix<T> = Tensor2D<T>
