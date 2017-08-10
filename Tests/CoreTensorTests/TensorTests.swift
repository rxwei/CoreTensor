//
//  TensorTests.swift
//  CoreTensor
//
//  Copyright 2016-2017 Richard Wei.
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
@testable import CoreTensor

class CoreTensorTests: XCTestCase {

    func testIndexCalculation() {
        let shape: TensorShape = [1, 2, 3] // 6
        let index: TensorIndex = [0, 0, 1]
        let contIndex = index.contiguousIndex(in: shape)
        XCTAssertEqual(contIndex, 1)
    }

    func testAddressing() {
        let shape: TensorShape = [1, 2, 3]
        let tensor = Tensor<Int>(shape: shape, repeating: 0)
        XCTAssertEqual(tensor[].shape, shape)
        XCTAssertEqual(tensor[0].shape, shape.dropFirst())
        XCTAssertEqual(tensor[0, 1].shape, shape.dropFirst(2))
        XCTAssertEqual(tensor[0][1].shape, shape.dropFirst(2))
        XCTAssertEqual(tensor[0, 1, 2].shape, .scalar)
        XCTAssertEqual(tensor[0][1][2].shape, .scalar)
        for subTensor in tensor[0] {
            XCTAssertEqual(subTensor.shape, shape.dropFirst(2))
            XCTAssertEqual(subTensor.units, [0, 0, 0])
        }
    }

    func testInit() {
        let scalar = Tensor<Int>(shape: [], units: [1])
        XCTAssertEqual(scalar.shape, .scalar)

        let highScalar = scalar.reshaped(as: [1, 1, 1])!
        XCTAssertTrue(highScalar.shape ~ .scalar)

        let scalars = Tensor<Int>(scalarElementsIn: 0..<10)
        XCTAssertEqual(scalars.units, ContiguousArray((0..<10).map {$0}))
        for (i, scalar) in scalars.enumerated() {
            XCTAssertEqual(scalar.unitCount, 1)
            XCTAssertEqual(scalar.units.first, i)
            XCTAssertEqual(scalar.shape, .scalar)
        }
    }

    func testEquality() {
        let highTensor = Tensor<Int>(shape: [1, 4, 3], unitsIncreasingFrom: 0)
        let lowTensor = Tensor<Int>(shape: [4, 3], unitsIncreasingFrom: 0)
        XCTAssertTrue(highTensor.unitsEqual(lowTensor))
        XCTAssertFalse(highTensor.elementsEqual(lowTensor))
        XCTAssertTrue(highTensor.shape ~ lowTensor.shape)
        XCTAssertTrue(highTensor.isSimilar(to: lowTensor))
        XCTAssertFalse(highTensor.isIsomorphic(to: lowTensor))

        let scalar = Tensor<Int>(shape: [], units: [1])
        let highScalar = Tensor<Int>(shape: [1, 1, 1], units: [100])
        XCTAssertTrue(highScalar.isSimilar(to: scalar))
        XCTAssertTrue(highScalar[0].shape ~ .scalar)
        XCTAssertFalse(highScalar.isIsomorphic(to: scalar))
        XCTAssertTrue(highScalar[0].shape ~ .scalar)
    }

    func testTextOutput() {
        let rank1: Tensor<Int> = Tensor(shape: [5], units: [1, 2, 3, 4, 5])
        let rank2 = Tensor<Int>(shape: [2, 3], units: [1, 2, 3,
                                                       4, 5, 6])
        let rank3 = Tensor<Int>(shape: [2, 3, 2], units: [1, 2, 3, 4, 5, 6,
                                                          7, 8, 9, 10, 11, 12])
        XCTAssertEqual("\(rank1)", "[1, 2, 3, 4, 5]")
        XCTAssertEqual("\(rank2)", "[[1, 2, 3], [4, 5, 6]]")
        XCTAssertEqual("\(rank3)", "[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]")
    }

    func testMutating() {
        var tensor = Tensor<Int>(shape: [5, 4, 3], repeating: 1)
        for i in 0..<tensor.units.count {
            tensor.multiplyUnit(at: i, by: i)
            tensor.incrementUnit(at: i, by: 1)
        }
        XCTAssertEqual(tensor.units, ContiguousArray((1...60).map {$0}))
    }

    func testAssignment() {
        var matrix = Tensor<Int>(shape: [4, 3], unitsIncreasingFrom: 0)
        let m2 = Tensor<Int>(shape: [2, 3], units: [10, 20, 30,
                                                    40, 50, 60])
        matrix[0, 0] = m2[1, 1]
        XCTAssertEqual(matrix.unit(at: [0, 0]), 50)
    }

    func testTranspose() {
        var matrix = Tensor<Int>(shape: [2, 3], units: [1, 2, 3, 4, 5, 6])
        var trans = Tensor<Int>(shape: matrix.shape.transpose, repeating: 0)
        for i in 0..<matrix.shape[0] {
            for j in 0..<matrix.shape[1] {
                trans[j, i] = matrix[i, j]
            }
        }
        XCTAssertEqual(trans.units, [1, 4, 2, 5, 3, 6])
    }

    func testSlice() {
        let tensor = Tensor<Int>(shape: [3, 4, 5], unitsIncreasingFrom: 0)

        /// Test shapes
        XCTAssertEqual(tensor[0].shape, [4, 5])
        XCTAssertEqual(tensor[0].elementShape, [5])
        XCTAssertEqual(tensor[0][0].shape, [5])
        XCTAssertEqual(tensor[0][0].elementShape, [])
        XCTAssertEqual(tensor[0][0][0].shape, [])
        XCTAssertEqual(tensor[0][0][0].elementShape, nil)
        XCTAssertEqual(tensor[0..<2].shape, [2, 4, 5])
        XCTAssertEqual(tensor[0..<2].elementShape, [4, 5])
        XCTAssertEqual(tensor[0][0..<3].shape, [3, 5])
        XCTAssertEqual(tensor[0][0..<3].elementShape, [5])

        /// Test element tensor indexing
        XCTAssertEqual(Array(tensor[0].units), Array(0..<20))
        XCTAssertEqual(Array(tensor[2].units), Array(40..<60))
        XCTAssertEqual(Array(tensor[0][0].units), Array(0..<5))
        XCTAssertEqual(Array(tensor[1][3].units), Array(35..<40))
        XCTAssertEqual(Array(tensor[2][0][3].units), [43])
        XCTAssertEqual(Array(tensor[1][3][2].units), [37])

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
    }

    static var allTests: [(String, (CoreTensorTests) -> () throws -> Void)] {
        return [
            ("testIndexCalculation", testIndexCalculation),
            ("testAddressing", testAddressing),
            // ("testInit", testInit),
            ("testTextOutput", testTextOutput),
            ("testEquality", testEquality),
            ("testMutating", testMutating),
            ("testAssignment", testAssignment),
            ("testTranspose", testTranspose),
            ("testSlice", testSlice),
        ]
    }

}
