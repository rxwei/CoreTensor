//
//  Slice.swift
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

import struct CoreTensor.TensorIndex
import struct CoreTensor.TensorShape
import struct CoreTensor.TensorSlice

/// Ranked tensor slice
public struct RankedTensorSlice<R: StaticRank> {
    public typealias DataType = R.DataType
    public typealias Shape = R.Shape
    public typealias ElementTensor = R.ElementTensor
    public typealias Base = TensorSlice<DataType>

    public private(set) var base: Base
    private let baseIndices: [Int]
    private let bounds: CountableRange<Int>?

    /// Tensor rank
    public static var rank: UInt {
        return R.rank
    }

    /// Tensor shape
    public var shape: Shape {
        let baseShapeArray = base.shape
        switch (Shape.self) {
        case is Shape1D.Type:
            return arrayToShape1D(Array(baseShapeArray.suffix(1))) as! Shape
        case is Shape2D.Type:
            return arrayToShape2D(Array(baseShapeArray.suffix(2))) as! Shape
        case is Shape3D.Type:
            return arrayToShape3D(Array(baseShapeArray.suffix(3))) as! Shape
        case is Shape4D.Type:
            return arrayToShape4D(Array(baseShapeArray.suffix(4))) as! Shape
        default:
            fatalError("Invalid shape")
        }
    }

    /// The number of items (atom) in the tensor.
    public var unitCount: Int {
        return units.count
    }

    /// Capacity reserved for sub-tensor
    public var capacity: Int {
        return units.capacity / unitCountPerElement
    }
}

public extension RankedTensorSlice {
    /// Indexing depth of this slice, i.e. the rank difference between the base
    /// and the slice
    private var indexingDepth: Int {
        return baseIndices.count
    }

    /// The number of units per element
    var unitCountPerElement: Int {
        return base.unitCountPerElement
    }

    var units: ArraySlice<DataType> {
        return base.units
    }

    var indices: CountableRange<Int> {
        return base.indices
    }

    var startIndex: Int {
        return base.startIndex
    }

    var endIndex: Int {
        return base.endIndex
    }
}

public extension RankedTensorSlice {

    init<A>(base: RankedTensor<A>, bounds: CountableRange<Int>?)
        where A.DataType == DataType {
            if let bounds = bounds {
                precondition(base.indices ~= bounds.startIndex
                    && base.indices ~= bounds.endIndex - 1,
                             "Slice is out of bounds")
            }
            self.baseIndices = []
            self.bounds = bounds
            self.base = TensorSlice<DataType>(base: base.storage, bounds: bounds)
    }

    init<A>(base: RankedTensorSlice<A>, bounds: CountableRange<Int>?)
        where A.DataType == DataType {
            if let bounds = bounds {
                precondition(base.indices ~= bounds.startIndex
                    && base.indices ~= bounds.endIndex - 1,
                             "Slice is out of bounds")
            }
            self.baseIndices = base.baseIndices
            self.bounds = bounds
            self.base = TensorSlice<DataType>(base: base.base, bounds: bounds)
    }

    init<A>(base: RankedTensor<A>, index: Int)
        where A.DataType == DataType {
            precondition(base.indices ~= index,
                         "Element index is out of bounds")
            self.baseIndices = [index]
            self.bounds = nil
            self.base = TensorSlice<DataType>(base: base.storage, index: index)
    }

    init<A>(base: RankedTensorSlice<A>, index: Int)
        where A.DataType == DataType {
            precondition(base.indices ~= index,
                         "Element index is out of bounds")
            self.baseIndices = base.baseIndices + [index]
            self.bounds = nil
            self.base = TensorSlice<DataType>(base: base.base, index: index)
    }

    init<A>(base: RankedTensor<A>, indices: [Int])
        where A.DataType == DataType {
            precondition(zip(RankedTensor<A>.shapeToArray(base.shape), indices)
                .filter{ !($1 >= 0 && $1 < $0) }.count == 0,
                         "Element indices are out of bounds")
            self.baseIndices = indices
            self.bounds = nil
            self.base = TensorSlice<DataType>(base: base.storage, indices: indices)
    }

    init<A>(base: RankedTensorSlice<A>, indices: [Int])
        where A.DataType == DataType {
            precondition(zip(RankedTensorSlice<A>.shapeToArray(base.shape), indices)
                .filter{ !($1 >= 0 && $1 < $0) }.count == 0,
                         "Element indices are out of bounds")
            self.baseIndices = base.baseIndices + indices
            self.bounds = nil
            self.base = TensorSlice<DataType>(base: base.base, indices: indices)
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    public init(shape: R.Shape, units: ContiguousArray<DataType>) {
        let base = RankedTensor<R>(shape: shape, units: units)
        self.init(base: base, bounds: base.indices)
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: R.Shape, repeating repeatedValue: DataType) {
        let contiguousSize: Int = RankedTensorSlice.shapeToArray(shape).reduce(1, *)
        let units = ContiguousArray(repeating: repeatedValue, count: contiguousSize)
        self.init(shape: shape, units: units)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: R.Shape, supplier: () -> DataType) {
        let contiguousSize: Int = RankedTensorSlice.shapeToArray(shape).reduce(1, *)
        let units = ContiguousArray((0..<contiguousSize).map { _ in supplier() })
        self.init(shape: shape, units: units)
    }

}

public extension RankedTensorSlice {
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
        return base.shape
    }
}

/// - TODO: Add conditional expressibility conformance in Swift 4

public extension RankedTensorSlice where R.DataType : Equatable {
    public static func ==<A>(lhs: RankedTensorSlice, rhs: RankedTensorSlice<A>) -> Bool
        where A.DataType == DataType {
            return lhs.elementsEqual(rhs)
    }

    public func elementsEqual<A>(_ other: RankedTensorSlice<A>) -> Bool
        where A.DataType == DataType {
            guard dynamicShape == other.dynamicShape else { return false }
            return units.elementsEqual(other.units)
    }

    public func unitsEqual<A>(_ other: RankedTensorSlice<A>) -> Bool
        where A.DataType == DataType {
            return units.elementsEqual(other.units)
    }
}

public extension RankedTensorSlice {
    public func isSimilar<A>(to other: RankedTensorSlice<A>) -> Bool
        where A.DataType == DataType {
            return dynamicShape.isSimilar(to: other.dynamicShape)
    }

    public func isIsomorphic<A>(to other: RankedTensorSlice<A>) -> Bool
        where A.DataType == DataType {
            return dynamicShape == other.dynamicShape
    }
}

extension RankedTensorSlice where R.DataType : Strideable {
    public init(shape: Shape, unitsIncreasingFrom lowerBound: DataType) {
        var unit = lowerBound
        self.init(shape: shape, supplier: {
            defer { unit = unit.advanced(by: 1) }
            return unit
        })
    }
}

extension RankedTensorSlice where R.DataType : Strideable, R.DataType.Stride : SignedInteger, R.Shape == (UInt) {
    public init(scalarElementsIn bounds: CountableRange<DataType>) {
        self.init(shape: (UInt(bounds.count)), units: ContiguousArray(bounds))
    }

    public init(scalarElementsIn bounds: CountableClosedRange<DataType>) {
        self.init(shape: (UInt(bounds.count)), units: ContiguousArray(bounds))
    }
}

public extension RankedTensorSlice {
    func unitIndex(fromIndex index: Int) -> Int {
        return unitCountPerElement * index
    }

    func unitSubrange(from tensorSubrange: Range<Int>) -> Range<Int> {
        return unitIndex(fromIndex: tensorSubrange.lowerBound)
            ..< unitIndex(fromIndex: tensorSubrange.upperBound)
    }
}

extension RankedTensorSlice : RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = ElementTensor
    // public typealias SubSequence = ElementTensor

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
                let unitIndex = index.advanced(by: units.startIndex)
                return units[unitIndex] as! Element
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
                base[index] = TensorSlice(scalar: scalar)
            case let tensor as TensorSlice1D<DataType>:
                base[index] = tensor.base
            case let tensor as TensorSlice2D<DataType>:
                base[index] = tensor.base
            case let tensor as TensorSlice3D<DataType>:
                base[index] = tensor.base
            default:
                fatalError("Invalid tensor slice type")
            }
        }
    }
}

public typealias TensorSlice1D<T> = RankedTensorSlice<R1<T>>
public typealias TensorSlice2D<T> = RankedTensorSlice<R2<T>>
public typealias TensorSlice3D<T> = RankedTensorSlice<R3<T>>

public typealias VectorSlice<T> = TensorSlice1D<T>
public typealias MatrixSlice<T> = TensorSlice2D<T>
