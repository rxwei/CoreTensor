//
//  TensorTests.swift
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

import XCTest
@testable import RankedTensor

class RankedTensorTests: XCTestCase {

    func testInit() {
        let vector = Vector<Int>(shape: (3), repeating: 5)
        XCTAssertTrue(vector.shape == (3))
        XCTAssertEqual(vector.units, ContiguousArray(repeating: 5, count: 3))

        let matrix = Matrix<Int>(shape: (4, 5), unitsIncreasingFrom: 0)
        XCTAssertTrue(matrix.shape == (4, 5))
        XCTAssertEqual(matrix.units, ContiguousArray((0..<20)))
    }

    func testRanks() {
        let tensor = Tensor3D<Int>(shape: (1, 2, 3), repeating: 0)
        XCTAssertTrue(tensor.shape == (1, 2, 3))
        XCTAssertEqual(tensor.shape.0, 1)
        XCTAssertEqual(tensor.shape.1, 2)
        XCTAssertEqual(tensor.shape.2, 3)
    }

    func testAddressing() {
        var tensor = Tensor3D<Int>(shape: (1, 2, 3), repeating: 0)

        /// Test subscript getters
        XCTAssertTrue(tensor[0].shape == (2, 3))
        XCTAssertTrue(tensor[0][1].shape == (3))
        for subTensor in tensor[0] {
            XCTAssertTrue(subTensor.shape == (3))
            XCTAssertEqual(subTensor.units, [0, 0, 0])
        }
        for unit in tensor[0][1] {
            XCTAssertEqual(unit, 0)
        }

        /// Test HOFs
        XCTAssertEqual(tensor[0].map { $0.shape }, [3, 3])
        XCTAssertEqual(tensor[0][1].map { $0 + 1 }, [1, 1, 1])

        /// Test subscript setters
        tensor[0] = TensorSlice2D<Int>(shape: (2, 3), repeating: 2)
        tensor[0][0] = TensorSlice1D<Int>(shape: (3), repeating: 1)
        tensor[0][0][0] = 0
        XCTAssertEqual(tensor.units, [0, 1, 1, 2, 2, 2])
    }

    func testEquality() {
        let highTensor = Tensor3D<Int>(shape: (1, 4, 3), repeating: 0)
        let lowTensor = Tensor2D<Int>(shape: (4, 3), repeating: 0)
        XCTAssertTrue(highTensor.unitsEqual(lowTensor))
        XCTAssertFalse(highTensor.elementsEqual(lowTensor))
        XCTAssertFalse(highTensor == lowTensor)
        XCTAssertFalse(highTensor.isIsomorphic(to: lowTensor))
    }

    func testMutating() {
        var tensor = Tensor3D<Int>(shape: (5, 4, 3), repeating: 1)
        for i in 0..<tensor.units.count {
            tensor.multiplyUnit(at: i, by: i)
            tensor.incrementUnit(at: i, by: 1)
        }
        XCTAssertEqual(tensor.units, ContiguousArray((1...60)))
    }
    
    func testSlicing() {
        var tensor = Tensor3D<Int>(shape: (3, 4, 5), unitsIncreasingFrom: 0)

        /// Test shapes
        XCTAssertTrue(tensor[0].shape == (4, 5))
        XCTAssertTrue(tensor[0][0].shape == (5))
        XCTAssertTrue(tensor[0..<2].shape == (2, 4, 5))
        XCTAssertTrue(tensor[0][0..<3].shape == (3, 5))

        /// Test element tensor indexing
        XCTAssertEqual(Array(tensor[0].units), Array(0..<20))
        XCTAssertEqual(Array(tensor[2].units), Array(40..<60))
        XCTAssertEqual(Array(tensor[0][0].units), Array(0..<5))
        XCTAssertEqual(Array(tensor[1][3].units), Array(35..<40))
        XCTAssertEqual(tensor[2][0][3], 43)
        XCTAssertEqual(tensor[1][3][2], 37)

        /// Test subtensor indexing
        XCTAssertEqual(Array(tensor[2..<3].units), Array(40..<60))
        XCTAssertEqual(Array(tensor[1][0..<2].units), Array(20..<30))
        XCTAssertEqual(Array(tensor[2][1..<2].units), Array(45..<50))
        XCTAssertEqual(Array(tensor[0][0][3..<5].units), Array(3..<5))

        /// Test subtensor collection properties
        XCTAssertTrue(tensor[0..<2].map {Array($0.units)}
            .elementsEqual([Array(0..<20), Array(20..<40)], by: ==))
        XCTAssertTrue(tensor[1][0..<2].map {Array($0.units)}
            .elementsEqual([Array(20..<25), Array(25..<30)], by: ==))

        /// Test subscript setters
        tensor[0] = TensorSlice2D<Int>(shape: (4, 5), repeating: 1)
        XCTAssertEqual(Array(tensor.units),
                       Array(repeating: 1, count: 20) + Array(20..<60))
        tensor[0..<2] = TensorSlice3D<Int>(shape: (2, 4, 5), unitsIncreasingFrom: 0)
        XCTAssertEqual(Array(tensor.units), Array(0..<60))
        tensor[0][1..<3] = TensorSlice2D<Int>(shape: (2, 5), unitsIncreasingFrom: 0)
        XCTAssertEqual(Array(tensor.units),
                       Array((0..<5)) + Array((0..<10)) + Array(15..<60))
        for scalarIndex in tensor[0][0].indices {
            tensor[0][0][scalarIndex] = scalarIndex - 5
        }
        XCTAssertEqual(Array(tensor.units), Array((-5..<10)) + Array(15..<60))
    }

    static var allTests: [(String, (RankedTensorTests) -> () throws -> Void)] {
        return [
            ("testInit", testInit),
            ("testRanks", testRanks),
            ("testAddressing", testAddressing),
            ("testEquality", testEquality),
            ("testMutating", testMutating),
            ("testSlicing", testSlicing),
        ]
    }

}
