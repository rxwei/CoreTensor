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

public struct Tensor<R : StaticRank> {
    public typealias Element = R.ElementTensor
    public typealias Shape = R.Shape
    public typealias DataType = R.DataType

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
    fileprivate init(shape: Shape, units: ContiguousArray<DataType>) {
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

    var dynamicShape: TensorShape {
        return TensorShape(Tensor.shapeToArray(self.shape))
    }
}

public typealias Vector<T> = Tensor<R1<T>>
public typealias Matrix<T> = Tensor<R2<T>>
public typealias Tensor1D<T> = Tensor<R1<T>>
public typealias Tensor2D<T> = Tensor<R2<T>>
public typealias Tensor3D<T> = Tensor<R3<T>>
public typealias Tensor4D<T> = Tensor<R4<T>>
