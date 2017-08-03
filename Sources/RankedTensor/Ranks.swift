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

public protocol TensorDataType {
    /// - TODO: Add dynamic (DLVM IR) type getters
    /// as requirements
    static var base: Self { get }
}

extension TensorDataType where Self : ExpressibleByIntegerLiteral {
    public static var base: Self {
        return 0
    }
}

extension Int8 : TensorDataType {}
extension Int16 : TensorDataType {}
extension Int32 : TensorDataType {}
extension Int64 : TensorDataType {}
extension Int : TensorDataType {}
extension Float : TensorDataType {}
extension Double : TensorDataType {}

public protocol StaticRank {
    associatedtype DataType : TensorDataType
    associatedtype Shape
    associatedtype ElementTensor
    associatedtype ElementShape
    associatedtype ElementRank : StaticRank // where ElementRank.DataType == DataType // , ElementRank.DataType : TensorDataType
    static var rank: UInt { get }
}

public struct R0<T : TensorDataType> : StaticRank {
    public typealias DataType = T
    public typealias Shape = ()
    public typealias ElementTensor = T
    public typealias ElementShape = ()
    public typealias ElementRank = R0
    public static var rank: UInt { return 0 }
}

public struct R1<T : TensorDataType> : StaticRank {
    public typealias DataType = T
    public typealias Shape = (UInt)
    public typealias ElementTensor = T
    public typealias ElementShape = ()
    // public typealias ElementRank = R0<T>
    public typealias ElementRank = R0

    public static var rank: UInt { return 1 }
}

public struct R2<T : TensorDataType> : StaticRank {
    public typealias DataType = T
    public typealias Shape = (UInt, UInt)
    public typealias ElementTensor = Tensor<R1<T>>
    public typealias ElementShape = (UInt)
    public typealias ElementRank = R1
    public static var rank: UInt { return 2 }
}

public struct R3<T : TensorDataType> : StaticRank {
    public typealias DataType = T
    public typealias Shape = (UInt, UInt, UInt)
    public typealias ElementTensor = Tensor<R2<T>>
    public typealias ElementShape = (UInt, UInt)
    public typealias ElementRank = R2
    public static var rank: UInt { return 3 }
}

public struct R4<T : TensorDataType> : StaticRank {
    public typealias DataType = T
    public typealias Shape = (UInt, UInt, UInt, UInt)
    public typealias ElementTensor = Tensor<R3<T>>
    public typealias ElementShape = (UInt, UInt, UInt)
    public typealias ElementRank = R3
    public static var rank: UInt { return 4 }
}

public protocol NonScalarRank {}
extension R1 : NonScalarRank {}
extension R2 : NonScalarRank {}
extension R3 : NonScalarRank {}
extension R4 : NonScalarRank {}
