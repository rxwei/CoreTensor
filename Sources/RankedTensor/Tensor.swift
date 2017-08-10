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

public struct Tensor<R: StaticRank> {
    public typealias DataType = R.DataType
    public typealias Shape = R.Shape
    public typealias ElementTensor = R.ElementTensor

    /// Tensor rank
    public static var rank: UInt {
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
    public fileprivate(set) var units: ContiguousArray<DataType>

    /// Capacity reserved for sub-tensor
    public var capacity: Int {
        return units.capacity / unitCountPerElement
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    public init(shape: Shape, units: ContiguousArray<DataType>) {
        // TODO: Reduce number of calls to `shapeToArray`
        let shapeArray = Tensor.shapeToArray(shape)
        let contiguousSize: Int = shapeArray.reduce(1, *)
        precondition(units.count >= contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.shape = shape
        self.units = ContiguousArray(units)
        self.unitCountPerElement = shapeArray.count == 0 || shapeArray[0] == 0 ?
            contiguousSize : contiguousSize / shapeArray[0]
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: Shape, repeating repeatedValue: DataType) {
        let contiguousSize: Int = Tensor.shapeToArray(shape).reduce(1, *)
        let units = ContiguousArray(repeating: repeatedValue, count: contiguousSize)
        self.init(shape: shape, units: units)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: Shape, supplier: () -> DataType) {
        let contiguousSize: Int = Tensor.shapeToArray(shape).reduce(1, *)
        let units = ContiguousArray((0..<contiguousSize).map { _ in supplier() })
        self.init(shape: shape, units: units)
    }
}

public extension Tensor {
    fileprivate static func shapeToArray(_ shape: Shape) -> [Int] {
        var shape = shape
        return withUnsafePointer(to: &shape) { ptr in
            ptr.withMemoryRebound(to: UInt.self, capacity: 1) { ptr in
                let buf = UnsafeBufferPointer(start: ptr, count: Int(R.rank))
                return buf.lazy.map {Int($0)}
            }
        }
    }

    var dynamicShape: TensorShape {
        return TensorShape(Tensor.shapeToArray(shape))
    }
}

public extension Tensor where R.Shape == Shape2D {
    fileprivate func getElementShape() -> Shape1D {
        return arrayToShape(Array(dynamicShape.dimensions.dropFirst()))
    }
}

/// - TODO: Add conditional expressibility conformance in Swift 4

public extension Tensor where R.DataType : Equatable {
    public static func ==<A>(lhs: Tensor<R>, rhs: Tensor<A>) -> Bool
        where A.DataType == DataType {
            return lhs.elementsEqual(rhs)
    }

    public func elementsEqual<A>(_ other: Tensor<A>) -> Bool
        where A.DataType == DataType {
            guard dynamicShape == other.dynamicShape else { return false }
            return units.elementsEqual(other.units)
    }

    public func unitsEqual<A>(_ other: Tensor<A>) -> Bool
        where A.DataType == DataType {
            return units.elementsEqual(other.units)
    }
}

public extension Tensor {
    public func isSimilar<A>(to other: Tensor<A>) -> Bool
        where A.DataType == DataType {
            return dynamicShape.isSimilar(to: other.dynamicShape)
    }

    public func isIsomorphic<A>(to other: Tensor<A>) -> Bool
        where A.DataType == DataType {
            return dynamicShape == other.dynamicShape
    }
}

extension Tensor where R.DataType : Strideable {
    public init(shape: Shape, unitsIncreasingFrom lowerBound: DataType) {
        var unit = lowerBound
        self.init(shape: shape, supplier: {
            defer { unit = unit.advanced(by: 1) }
            return unit
        })
    }
}

extension Tensor where R.DataType : Strideable, R.DataType.Stride : SignedInteger, R.Shape == (UInt) {
    public init(scalarElementsIn bounds: CountableRange<DataType>) {
        self.init(shape: (UInt(bounds.count)), units: ContiguousArray(bounds))
    }

    public init(scalarElementsIn bounds: CountableClosedRange<DataType>) {
        self.init(shape: (UInt(bounds.count)), units: ContiguousArray(bounds))
    }
}

public extension Tensor {
    func makeUnitIterator() -> IndexingIterator<ContiguousArray<DataType>> {
        return units.makeIterator()
    }

    mutating func updateUnit(at index: Int, to newValue: DataType) {
        units[units.startIndex.advanced(by: index)] = newValue
    }
}

public extension Tensor where R.DataType : Numeric {
    mutating func incrementUnit(at index: Int, by newValue: DataType) {
        units[units.startIndex.advanced(by: index)] += newValue
    }

    mutating func decrementUnit(at index: Int, by newValue: DataType) {
        units[units.startIndex.advanced(by: index)] -= newValue
    }

    mutating func multiplyUnit(at index: Int, by newValue: DataType) {
        units[units.startIndex.advanced(by: index)] *= newValue
    }
}

public extension Tensor where R.DataType : BinaryInteger {
    mutating func divideUnit(at index: Int, by newValue: DataType) {
        units[units.startIndex.advanced(by: index)] /= newValue
    }
}

public extension Tensor where R.DataType : FloatingPoint {
    mutating func divideUnit(at index: Int, by newValue: DataType) {
        units[units.startIndex.advanced(by: index)] /= newValue
    }
}

public extension Tensor {
    func unitIndex(fromIndex index: Int) -> Int {
        return unitCountPerElement * index
    }

    func unitSubrange(from tensorSubrange: Range<Int>) -> Range<Int> {
        return unitIndex(fromIndex: tensorSubrange.lowerBound)
            ..< unitIndex(fromIndex: tensorSubrange.upperBound)
    }
}

extension Tensor : RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = ElementTensor

    /// Get indices corresponding to the units of the element tensor at index
    fileprivate func getElementTensorRange(index: Int) -> CountableRange<Int> {
        let elementTensorShape = dynamicShape.dropFirst()
        let contiguousIndex = unitIndex(fromIndex: index)
        return contiguousIndex..<contiguousIndex+elementTensorShape.contiguousSize
    }

    /// Get units in the element tensor at index
    fileprivate func getElementTensorUnits(index: Int) -> ContiguousArray<DataType> {
        let range = getElementTensorRange(index: index)
        return ContiguousArray(units[range])
    }

    /// Access the scalar element or element tensor at index
    public subscript(index: Int) -> Element {
        get {
            switch (Element.self) {
            case is DataType.Type:
                return units[index] as! Element
            case is Tensor1D<DataType>.Type:
                let elementShape: Shape1D = arrayToShape(Array(dynamicShape.dimensions.dropFirst()))
                let elementUnits = getElementTensorUnits(index: index)
                return Tensor1D<DataType>(shape: elementShape, units: elementUnits) as! Element
            case is Tensor2D<DataType>.Type:
                let elementShape: Shape2D = arrayToShape(Array(dynamicShape.dimensions.dropFirst()))
                let elementUnits = getElementTensorUnits(index: index)
                return Tensor2D<DataType>(shape: elementShape, units: elementUnits) as! Element
            case is Tensor3D<DataType>.Type:
                let elementShape: Shape3D = arrayToShape(Array(dynamicShape.dimensions.dropFirst()))
                let elementUnits = getElementTensorUnits(index: index)
                return Tensor3D<DataType>(shape: elementShape, units: elementUnits) as! Element
            default:
                fatalError("Invalid element tensor type")
            }
        }
        set {
            switch (newValue) {
            case let scalar as DataType:
                units[index] = scalar
            case let tensor as Tensor1D<DataType>:
                units.replaceSubrange(getElementTensorRange(index: index), with: tensor.units)
            case let tensor as Tensor2D<DataType>:
                units.replaceSubrange(getElementTensorRange(index: index), with: tensor.units)
            case let tensor as Tensor3D<DataType>:
                units.replaceSubrange(getElementTensorRange(index: index), with: tensor.units)
            default:
                fatalError("Invalid element tensor type")
            }
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

public typealias Tensor1D<T> = Tensor<R1<T>>
public typealias Tensor2D<T> = Tensor<R2<T>>
public typealias Tensor3D<T> = Tensor<R3<T>>
public typealias Tensor4D<T> = Tensor<R4<T>>

public typealias Vector<T> = Tensor1D<T>
public typealias Matrix<T> = Tensor2D<T>
