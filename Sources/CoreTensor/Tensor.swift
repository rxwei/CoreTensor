//
//  Tensor.swift
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

public protocol TensorProtocol : RangeReplaceableCollection, RandomAccessCollection
    where IndexDistance == Int
{
    associatedtype UnitType
    var units: ContiguousArray<UnitType> { get }
    var unitCountPerElement: IndexDistance { get }
    var elementShape: TensorShape? { get }
    subscript(index: Index) -> Self { get }
    subscript(index: TensorIndex) -> Self { get }
}

public extension TensorProtocol {
    var shape: TensorShape {
        return elementShape?.prepending(count) ?? .scalar
    }

    var unitCount: Int {
        return units.count
    }

    func unit(at index: Int) -> UnitType {
        return units[units.startIndex.advanced(by: index)]
    }

    func unit(at index: TensorIndex) -> UnitType {
        return self[index].units[0]
    }
}

internal extension TensorProtocol {
    func unitIndex(fromIndex index: Int) -> Int {
        return unitCountPerElement * index
    }

    func unitSubrange(from tensorSubrange: Range<Int>) -> Range<Int> {
        return unitIndex(fromIndex: tensorSubrange.lowerBound)
           ..< unitIndex(fromIndex: tensorSubrange.upperBound)
    }
}

/// Tensor
public struct Tensor<UnitType> : TensorProtocol {
    public typealias Element = Tensor<UnitType>

    /// Sub-tensor (element) shape
    public internal(set) var elementShape: TensorShape? {
        didSet {
            unitCountPerElement = elementShape?.contiguousSize ?? 0
        }
    }

    public var isScalar: Bool {
        return elementShape == nil
    }

    /// The number of units (atoms) per element (sub-tensor).
    /// - Note: This holds the same value as the computed property
    /// `shape.contiguousSize`, which is extremely frequently used.
    /// We cache it in this variable whenever a new shape is set
    public private(set) var unitCountPerElement: Int

    /// Contiguous storage
    public fileprivate(set) var units: ContiguousArray<UnitType>

    /// Capacity reserved for sub-tensors
    public var capacity: Int {
        return units.capacity / unitCountPerElement
    }

    fileprivate init(elementShape: TensorShape?, units: ContiguousArray<UnitType>) {
        let elementContiguousSize = elementShape?.contiguousSize ?? 1
        precondition(units.count % elementContiguousSize == 0,
                     "Unit count does not match element shape")
        self.unitCountPerElement = elementShape == nil ? 0 : elementContiguousSize
        self.elementShape = elementShape
        self.units = ContiguousArray(units)
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    fileprivate init(shape: TensorShape, units: ContiguousArray<UnitType>) {
        precondition(units.count >= shape.contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.init(elementShape: shape.isScalar ? nil : shape.dropFirst(),
                  units: ContiguousArray(units.prefix(shape.contiguousSize)))
    }

    /// Initialize an empty tensor of scalar elements
    public init() {
        self.init(elementShape: .scalar)
    }

    /// Initialize an empty tensor
    public init(elementShape: TensorShape) {
        self.init(elementShape: elementShape, units: [])
    }

    /// Initialize a scalar tensor
    public init(scalar: UnitType) {
        self.init(elementShape: nil, units: [scalar])
    }

    /// Initialize a tensor from a sequence of tensors of element shape
    public init<S : Sequence>(elementShape: TensorShape, elements: S)
        where S.Iterator.Element == Tensor<UnitType> {
        self.init(elementShape: elementShape)
        self.append(contentsOf: elements)
    }

    /// Initialize a tensor from a collection of tensors of element shape
    public init<C : Collection>(elementShape: TensorShape, elements: C)
        where C.Iterator.Element == Tensor<UnitType>, C.IndexDistance == Int {
        self.init(elementShape: elementShape)
        self.append(contentsOf: elements)
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    public init<S: Sequence>(shape: TensorShape, units: S,
                             vacancySupplier supplier: (() -> UnitType)? = nil)
        where S.Iterator.Element == UnitType, S.SubSequence : Sequence,
              S.SubSequence.Iterator.Element == UnitType {
        let contiguousSize = shape.contiguousSize
        var slice = ContiguousArray(units.prefix(contiguousSize))
        /// If elements fewer than required by the shape and supplier is provided
        /// generate new elements using the supplier until vacancy is filled
        if slice.count < contiguousSize, let supplier = supplier {
            slice.reserveCapacity(shape.contiguousSize)
            slice.append(contentsOf: (0..<contiguousSize).map { _ in supplier() })
        }
        self.init(shape: shape, units: slice)
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: TensorShape, repeating repeatedValue: UnitType) {
        let units = ContiguousArray(repeating: repeatedValue, count: shape.contiguousSize)
        self.init(shape: shape, units: units)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: TensorShape, supplier: () -> UnitType) {
        let units = ContiguousArray((0..<shape.contiguousSize).map { _ in supplier() })
        self.init(shape: shape, units: units)
    }

}

/// - TODO: Add conditional expressibility conformance in Swift 4

public extension Tensor where UnitType : Equatable {

    public static func ==(lhs: Tensor<UnitType>, rhs: Tensor<UnitType>) -> Bool {
        return lhs.elementsEqual(rhs)
    }

    public func elementsEqual(_ other: Tensor<UnitType>) -> Bool {
        guard shape == other.shape else { return false }
        return shape.elementsEqual(other.shape) && units.elementsEqual(other.units)
    }

    public func unitsEqual(_ other: Tensor<UnitType>) -> Bool {
        return units.elementsEqual(other.units)
    }

}

public extension TensorProtocol {

    public func isSimilar(to other: Self) -> Bool {
        return shape ~ other.shape
    }

    public func isIsomorphic(to other: Self) -> Bool {
        return shape == other.shape
    }

}

extension Tensor where UnitType : Strideable {

    public init(shape: TensorShape, unitsIncreasingFrom lowerBound: UnitType) {
        var unit = lowerBound
        self.init(shape: shape, supplier: {
            defer { unit = unit.advanced(by: 1) }
            return unit
        })
    }

}

extension Tensor where UnitType : Strideable, UnitType.Stride : SignedInteger {

    public init(scalarElementsIn bounds: CountableRange<UnitType>) {
        self.init(elementShape: .scalar, units: ContiguousArray(bounds))
    }

    public init(scalarElementsIn bounds: CountableClosedRange<UnitType>) {
        self.init(elementShape: .scalar, units: ContiguousArray(bounds))
    }

}

public extension Tensor {

    func makeUnitIterator() -> IndexingIterator<ContiguousArray<UnitType>> {
        return units.makeIterator()
    }

    mutating func updateUnit(at index: Int, to newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] = newValue
    }

}

public extension Tensor where UnitType : Numeric {

    mutating func incrementUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] += newValue
    }

    mutating func decrementUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] -= newValue
    }

    mutating func multiplyUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] *= newValue
    }

}

public extension Tensor where UnitType : BinaryInteger {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] /= newValue
    }
}

public extension Tensor where UnitType : FloatingPoint {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        units[units.startIndex.advanced(by: index)] /= newValue
    }
}

internal extension RangeReplaceableRandomAccessSlice {
    var baseRange: Range<Base.Index> {
        return startIndex..<endIndex
    }
}

public typealias TensorSlice<UnitType> =
    RangeReplaceableRandomAccessSlice<Tensor<UnitType>>

extension Tensor : RangeReplaceableCollection {

    public typealias SubSequence = TensorSlice<UnitType>

    public mutating func append(_ newElement: Tensor<UnitType>) {
        precondition(newElement.shape == elementShape, "Element shape mismatch")
        reserveCapacity(count + 1)
        units.append(contentsOf: newElement.units)
    }

    public mutating func reserveCapacity(_ minimumCapacity: Int) {
        let cap = Swift.max(units.capacity, unitCountPerElement * minimumCapacity)
        units.reserveCapacity(cap)
    }

    @discardableResult
    public mutating func remove(at index: Int) -> Tensor<UnitType> {
        precondition(indices.contains(index), "Index out of range")
        let range = unitSubrange(from: index..<index+1)
        defer { units.removeSubrange(range) }
        return Tensor(self[range])
    }

    public mutating func append<S : Sequence>(contentsOf newElements: S)
        where S.Iterator.Element == Tensor<UnitType> {
        for element in newElements {
            append(element)
        }
    }

    public mutating func append<C : Collection>(contentsOf newElements: C)
        where C.Iterator.Element == Tensor<UnitType>, C.IndexDistance == Int {
        reserveCapacity(count + newElements.count)
        for element in newElements {
            append(element)
        }
    }

    public mutating func append(contentsOf newElements: Tensor<UnitType>) {
        precondition(newElements.elementShape == elementShape, "Element shape mismatch")
        reserveCapacity(count + newElements.count)
        units.append(contentsOf: newElements.units)
    }

    public mutating func replaceSubrange<C : Collection>
        (_ subrange: Range<Int>, with newElements: C) where C.Iterator.Element == Tensor<UnitType> {
        let range = unitSubrange(from: subrange)
        let elemShape = elementShape
        units.replaceSubrange(range, with: newElements.lazy.flatMap { elem -> ContiguousArray<UnitType> in
            precondition(elemShape == elem.shape, "Element shape mismatch")
            return elem.units
        })
    }

    public subscript(bounds: Range<Int>) -> TensorSlice<UnitType> {
        get {
            precondition(isScalar,
                         "I am a scalar and I have no dimensions!")
            return RangeReplaceableRandomAccessSlice(base: self, bounds: bounds)
        }
        set {
            precondition(isScalar,
                         "I am a scalar and I have no dimensions!")
            precondition(newValue.base.elementShape == elementShape,
                         "Element shape mismatch")
            units[unitSubrange(from: bounds)] =
                newValue.base.units[newValue.base.unitSubrange(from: newValue.baseRange)]
        }
    }

}

extension Tensor : RandomAccessCollection {
    public typealias Index = Int

    /// Access a sub-tensor at index
    public subscript(index: TensorIndex) -> Tensor<UnitType> {
        get {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize as Range
            return Tensor(shape: newShape, units: units[range])
        }
        set {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            precondition(newShape == newValue.shape, "Shape mismatch")
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            units.replaceSubrange(range, with: newValue.units)
        }
    }

    /// Access a sub-tensor at an index specified by a list of dimensional indices
    /// - parameter indices: tensor indices
    /// - note: the count of indices must equal the raw rank of the tensor
    public subscript(indices: Int...) -> Tensor<UnitType> {
        get {
            return self[TensorIndex(indices)]
        }
        set {
            self[TensorIndex(indices)] = newValue
        }
    }

    /// Access a sub-tensor at the current dimension at index
    public subscript(index: Int) -> Tensor<UnitType> {
        get {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst()
            let contiguousIndex = unitIndex(fromIndex: index)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            return Tensor(shape: newShape, units: units[range])
        }
        set {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst()
            let contiguousIndex = unitIndex(fromIndex: index)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            units.replaceSubrange(range, with: newValue.units)
        }
    }

    public var count: Int {
        return isScalar ? 0 : units.count / unitCountPerElement
    }

    /// Returns a sequence of tensor indices for scalar elements
    public var indices: CountableRange<Int> {
        return 0..<count
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

public extension RangeReplaceableRandomAccessSlice
    where Base : TensorProtocol, Base.Index == Int
{
    public var shape: TensorShape {
        return base.elementShape?.prepending(count) ?? .scalar
    }

    public var units: ArraySlice<Base.UnitType> {
        return base.units[base.unitSubrange(from: baseRange)]
    }

    var unitCount: Int {
        return count * base.unitCountPerElement
    }

}

public extension Tensor {

    public func reshaped(as newShape: TensorShape) -> Tensor? {
        guard self.shape.contiguousSize == newShape.contiguousSize else {
            return nil
        }
        return Tensor(shape: newShape, units: units)
    }

}

public extension Tensor {

    func withUnsafeBufferPointer<Result>
        (_ body: (UnsafeBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try units.withUnsafeBufferPointer { ptr in
            try body(ptr)
        }
    }

    mutating func withUnsafeMutableBufferPointer<Result>
        (_ body: (inout UnsafeMutableBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try units.withUnsafeMutableBufferPointer { ptr in
            try body(&ptr)
        }
    }

}

extension Tensor : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write(
            isScalar ? String(describing: units[0])
                     : "[\(map{"\($0)"}.joined(separator: ", "))]"
        )
    }
}
