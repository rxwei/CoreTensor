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

    public static var rank: UInt {
        return R.rank
    }

    public var shape: Shape

    /// - TODO: Add stuff here
}

public extension Tensor {
    var dynamicShape: TensorShape {
        var shape = self.shape
        return withUnsafePointer(to: &shape) { ptr in
            ptr.withMemoryRebound(to: UInt.self, capacity: 1) { ptr in
                let buf = UnsafeBufferPointer(start: ptr, count: Int(R.rank))
                return TensorShape(buf.lazy.map{Int($0)})
            }
        }
    }
}

public typealias Tensor1D<T> = Tensor<R1<T>>
public typealias Tensor2D<T> = Tensor<R2<T>>
public typealias Tensor3D<T> = Tensor<R3<T>>
public typealias Tensor4D<T> = Tensor<R4<T>>
