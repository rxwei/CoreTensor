//
//  Tensor.swift
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

import struct CoreTensor.TensorShape

public struct Tensor<R : StaticRank, DataType : TensorDataType> {
    public typealias Shape = R.Shape
    public typealias ElementShape = R.ElementShape
    public typealias ElementRank = R.ElementRank

    /// Tensor rank
    public static var rank: UInt {
        return R.rank
    }

    /// Tensor shape
    public var shape: Shape

    /// The number of items (atoms) per element (sub-tensor).
    /// - Note: This holds the same value as the computed property
    /// `shape.contiguousSize`, which is extremely frequently used.
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
        self.unitCountPerElement = contiguousSize / shapeArray[0]
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
                return buf.lazy.map{Int($0)}
            }
        }
    }

    fileprivate static func arrayToElementShape(_ array: [Int]) -> ElementShape {
        var array = array
        return withUnsafePointer(to: &array[0]) { ptr in
            ptr.withMemoryRebound(to: ElementShape.self, capacity: 1) { ptr in
                return ptr.pointee
            }
        }
    }

    var dynamicShape: TensorShape {
        return TensorShape(Tensor.shapeToArray(self.shape))
    }
}

/// - TODO: Add conditional expressibility conformance in Swift 4

public extension Tensor where DataType : Equatable {
    public static func ==<A>(lhs: Tensor<R, DataType>, rhs: Tensor<A, DataType>) -> Bool {
        return lhs.elementsEqual(rhs)
    }

    public func elementsEqual<A>(_ other: Tensor<A, DataType>) -> Bool {
            // guard shape == other.shape else { return false }
            guard dynamicShape == other.dynamicShape else { return false }
            return units.elementsEqual(other.units)
    }

    public func unitsEqual<A>(_ other: Tensor<A, DataType>) -> Bool {
            return units.elementsEqual(other.units)
    }
}

public extension Tensor {
    // TODO: Fix `isSimilar` bug ('~' is not a binary operator)
    /*
    public func isSimilar<A>(to other: Tensor<A>) -> Bool {
        return dynamicShape ~ other.dynamicShape
    }
    */

    public func isIsomorphic<A>(to other: Tensor<A, DataType>) -> Bool {
        return dynamicShape == other.dynamicShape
    }
}

extension Tensor where DataType : Strideable {
    public init(shape: Shape, unitsIncreasingFrom lowerBound: DataType) {
        var unit = lowerBound
        self.init(shape: shape, supplier: {
            defer { unit = unit.advanced(by: 1) }
            return unit
        })
    }
}

extension Tensor where DataType : Strideable, DataType.Stride : SignedInteger, R.Shape == (UInt) {
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

public extension Tensor where DataType : Numeric {
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

public extension Tensor where DataType : BinaryInteger {
    mutating func divideUnit(at index: Int, by newValue: DataType) {
        units[units.startIndex.advanced(by: index)] /= newValue
    }
}

public extension Tensor where DataType : FloatingPoint {
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

/// - TODO: Add conditional expressibility conformance in Swift 4
// `extension Tensor : RandomAccessCollection where R : NonScalarRank { ... }`

extension Tensor : RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = Tensor<ElementRank, DataType>

    /// Access a sub-tensor at the current dimension at index
    public subscript(index: Int) -> Element {
            get {
                let newTensorShape = dynamicShape.dropFirst()
                let contiguousIndex = unitIndex(fromIndex: index)
                let range = contiguousIndex..<contiguousIndex+newTensorShape.contiguousSize
                let newShape = Tensor.arrayToElementShape(Array(dynamicShape.dimensions.dropFirst()))
                return Element(shape: newShape, units: ContiguousArray(units[range]))
            }
            set {
                let newShape = dynamicShape.dropFirst()
                let contiguousIndex = unitIndex(fromIndex: index)
                let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
                units.replaceSubrange(range, with: newValue.units)
            }
    }

    public var count: Int {
        return units.count
    }

    /// Returns a sequence of tensor indices for scalar elements
    public var indices: CountableRange<Int> {
        return 0..<endIndex
    }

    public var startIndex: Int {
        return 0
    }

    public var endIndex: Int {
        return count / unitCountPerElement
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

extension Tensor where R.Shape == (UInt) {
    /// Access scalar element at index
    public subscript(index: Int) -> DataType {
        get {
            return units[index]
        }
        set {
            precondition(!units.isEmpty, "Vector is empty")
            units[0] = newValue
        }
    }
}

public typealias Tensor1D<T : TensorDataType> = Tensor<R1, T>
public typealias Tensor2D<T : TensorDataType> = Tensor<R2, T>
public typealias Tensor3D<T : TensorDataType> = Tensor<R3, T>
public typealias Tensor4D<T : TensorDataType> = Tensor<R4, T>

public typealias Vector<T : TensorDataType> = Tensor1D<T>
public typealias Matrix<T : TensorDataType> = Tensor2D<T>
