//
//  Ranks.swift
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

public protocol StaticRank {
    associatedtype DataType
    associatedtype Shape
    associatedtype ElementTensor
    static var rank: UInt { get }
}

public struct R1<T> : StaticRank {
    public typealias DataType = T
    public typealias Shape = Shape1D
    public typealias ElementTensor = T
    public static var rank: UInt { return 1 }
}

public struct R2<T> : StaticRank {
    public typealias DataType = T
    public typealias Shape = (UInt, UInt)
    public typealias ElementTensor = Tensor<R1<T>>
    public static var rank: UInt { return 2 }
}

public struct R3<T> : StaticRank {
    public typealias DataType = T
    public typealias Shape = (UInt, UInt, UInt)
    public typealias ElementTensor = Tensor<R2<T>>
    public static var rank: UInt { return 3 }
}

public struct R4<T> : StaticRank {
    public typealias DataType = T
    public typealias Shape = (UInt, UInt, UInt, UInt)
    public typealias ElementTensor = Tensor<R3<T>>
    public static var rank: UInt { return 4 }
}
