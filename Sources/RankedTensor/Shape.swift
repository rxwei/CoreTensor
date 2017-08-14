//
//  Shape.swift
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

public typealias Shape1D = (UInt)
public typealias Shape2D = (UInt, UInt)
public typealias Shape3D = (UInt, UInt, UInt)
public typealias Shape4D = (UInt, UInt, UInt, UInt)

internal func arrayToShape1D(_ array: [Int]) -> Shape1D {
    var array = array
    return withUnsafePointer(to: &array[0]) { ptr in
        ptr.withMemoryRebound(to: Shape1D.self, capacity: 1) { ptr in
            return ptr.pointee
        }
    }
}

internal func arrayToShape2D(_ array: [Int]) -> Shape2D {
    var array = array
    return withUnsafePointer(to: &array[0]) { ptr in
        ptr.withMemoryRebound(to: Shape2D.self, capacity: 1) { ptr in
            return ptr.pointee
        }
    }
}

internal func arrayToShape3D(_ array: [Int]) -> Shape3D {
    var array = array
    return withUnsafePointer(to: &array[0]) { ptr in
        ptr.withMemoryRebound(to: Shape3D.self, capacity: 1) { ptr in
            return ptr.pointee
        }
    }
}

internal func arrayToShape4D(_ array: [Int]) -> Shape4D {
    var array = array
    return withUnsafePointer(to: &array[0]) { ptr in
        ptr.withMemoryRebound(to: Shape4D.self, capacity: 1) { ptr in
            return ptr.pointee
        }
    }
}
