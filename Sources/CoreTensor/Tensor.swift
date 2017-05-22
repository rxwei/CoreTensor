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

public protocol TensorProtocol : RangeReplaceableCollection, RandomAccessCollection {
    associatedtype ItemType
    var items: ContiguousArray<ItemType> { get }
    var itemCountPerElement: IndexDistance { get }
    var elementShape: TensorShape? { get }
    subscript(index: Index) -> Self { get }
    subscript(index: TensorIndex) -> Self { get }
}

public extension TensorProtocol where IndexDistance == Int {
    /// Tensor shape
    var shape: TensorShape {
        return elementShape?.prepending(count) ?? .scalar
    }

    var itemCount: Int {
        return items.count
    }

    func item(at index: Int) -> ItemType {
        return items[items.startIndex.advanced(by: index)]
    }

    func item(at index: TensorIndex) -> ItemType {
        return self[index].items[0]
    }
}

internal extension TensorProtocol where IndexDistance == Int {
    func itemIndex(fromIndex index: Int) -> Int {
        return itemCountPerElement * index
    }

    func itemSubrange(from tensorSubrange: Range<Int>) -> Range<Int> {
        return itemIndex(fromIndex: tensorSubrange.lowerBound)
           ..< itemIndex(fromIndex: tensorSubrange.upperBound)
    }
}

/// Tensor
public struct Tensor<ItemType> : TensorProtocol {
    public typealias Element = Tensor<ItemType>

    /// Sub-tensor (element) shape
    public internal(set) var elementShape: TensorShape? {
        didSet {
            itemCountPerElement = elementShape?.contiguousSize ?? 0
        }
    }

    public var isScalar: Bool {
        return elementShape == nil
    }

    /// The number of items (atoms) per element (sub-tensor).
    /// - Note: This holds the same value as the computed property
    /// `shape.contiguousSize`, which is extremely frequently used.
    /// We cache it in this variable whenever a new shape is set
    public private(set) var itemCountPerElement: Int

    /// Contiguous storage
    public fileprivate(set) var items: ContiguousArray<ItemType>

    /// Capacity reserved for sub-tensors
    public var capacity: Int {
        return items.capacity / itemCountPerElement
    }

    fileprivate init(elementShape: TensorShape?, items: ContiguousArray<ItemType>) {
        let elementContiguousSize = elementShape?.contiguousSize ?? 1
        precondition(items.count % elementContiguousSize == 0,
                     "Item count does not match element shape")
        self.itemCountPerElement = elementShape == nil ? 0 : elementContiguousSize
        self.elementShape = elementShape
        self.items = ContiguousArray(items)
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    fileprivate init(shape: TensorShape, items: ContiguousArray<ItemType>) {
        precondition(items.count >= shape.contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.init(elementShape: shape.isScalar ? nil : shape.dropFirst(),
                  items: ContiguousArray(items.prefix(shape.contiguousSize)))
    }

    /// Initialize an empty tensor of scalar elements
    public init() {
        self.init(elementShape: .scalar)
    }

    /// Initialize an empty tensor
    public init(elementShape: TensorShape) {
        self.init(elementShape: elementShape, items: [])
    }

    /// Initialize a scalar tensor
    public init(scalar: ItemType) {
        self.init(elementShape: nil, items: [scalar])
    }

    /// Initialize a tensor from a sequence of tensors of element shape
    public init<S : Sequence>(elementShape: TensorShape, elements: S)
        where S.Iterator.Element == Tensor<ItemType> {
        self.init(elementShape: elementShape)
        self.append(contentsOf: elements)
    }

    /// Initialize a tensor from a collection of tensors of element shape
    public init<C : Collection>(elementShape: TensorShape, elements: C)
        where C.Iterator.Element == Tensor<ItemType>, C.IndexDistance == Int {
        self.init(elementShape: elementShape)
        self.append(contentsOf: elements)
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    public init<S: Sequence>(shape: TensorShape, items: S,
                             vacancySupplier supplier: (() -> ItemType)? = nil)
        where S.Iterator.Element == ItemType, S.SubSequence : Sequence,
              S.SubSequence.Iterator.Element == ItemType {
        let contiguousSize = shape.contiguousSize
        var slice = ContiguousArray(items.prefix(contiguousSize))
        /// If elements fewer than required by the shape and supplier is provided
        /// generate new elements using the supplier until vacancy is filled
        if slice.count < contiguousSize, let supplier = supplier {
            slice.reserveCapacity(shape.contiguousSize)
            slice.append(contentsOf: (0..<contiguousSize).map { _ in supplier() })
        }
        self.init(shape: shape, items: slice)
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    public init(shape: TensorShape, repeating repeatedValue: ItemType) {
        let items = ContiguousArray(repeating: repeatedValue, count: shape.contiguousSize)
        self.init(shape: shape, items: items)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: TensorShape, supplier: () -> ItemType) {
        let items = ContiguousArray((0..<shape.contiguousSize).map { _ in supplier() })
        self.init(shape: shape, items: items)
    }

}

/// - TODO: Add conditional expressibility conformance in Swift 4

// MARK: - Element equality and item equality
public extension Tensor where ItemType : Equatable {

    public static func ==(lhs: Tensor<ItemType>, rhs: Tensor<ItemType>) -> Bool {
        return lhs.elementsEqual(rhs)
    }

    public func elementsEqual(_ other: Tensor<ItemType>) -> Bool {
        guard shape == other.shape else { return false }
        return shape.elementsEqual(other.shape) && items.elementsEqual(other.items)
    }

    public func itemsEqual(_ other: Tensor<ItemType>) -> Bool {
        return items.elementsEqual(other.items)
    }

}

// MARK: - Isomorphism
public extension TensorProtocol where IndexDistance == Int {

    public func isSimilar(to other: Self) -> Bool {
        return shape ~ other.shape
    }

    public func isIsomorphic(to other: Self) -> Bool {
        return shape == other.shape
    }

}

// MARK: - Initializers for Strideable item type
extension Tensor where ItemType : Strideable {

    public init(shape: TensorShape, itemsIncreasingFrom lowerBound: ItemType) {
        var item = lowerBound
        self.init(shape: shape, supplier: {
            defer { item = item.advanced(by: 1) }
            return item
        })
    }

}

// MARK: - Initializers for Strideable item type
extension Tensor where ItemType : Strideable, ItemType.Stride : SignedInteger {

    public init(scalarElementsIn bounds: CountableRange<ItemType>) {
        self.init(elementShape: .scalar, items: ContiguousArray(bounds))
    }

    public init(scalarElementsIn bounds: CountableClosedRange<ItemType>) {
        self.init(elementShape: .scalar, items: ContiguousArray(bounds))
    }

}

// MARK: - Item mutation
public extension Tensor {

    func makeItemIterator() -> IndexingIterator<ContiguousArray<ItemType>> {
        return items.makeIterator()
    }

    mutating func updateItem(at index: Int, to newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] = newValue
    }

}

#if swift(>=4.0)

// MARK: - Numeric
public extension Tensor where ItemType : Numeric {

    mutating func incrementItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] += newValue
    }

    mutating func decrementItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] -= newValue
    }

    mutating func multiplyItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] *= newValue
    }

}

// MARK: - BinaryInteger
public extension Tensor where ItemType : BinaryInteger {
    mutating func divideItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] /= newValue
    }
}

#else // Swift 3

// MARK: - Numeric
public extension Tensor where ItemType : IntegerArithmetic {

    mutating func incrementItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] += newValue
    }

    mutating func decrementItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] -= newValue
    }

    mutating func multiplyItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] *= newValue
    }

    mutating func divideItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] /= newValue
    }

}

#endif

// MARK: - FloatingPoint
public extension Tensor where ItemType : FloatingPoint {

    mutating func incrementItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] += newValue
    }

    mutating func decrementItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] -= newValue
    }

    mutating func multiplyItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] *= newValue
    }

    mutating func divideItem(at index: Int, by newValue: ItemType) {
        items[items.startIndex.advanced(by: index)] /= newValue
    }

}

internal extension RangeReplaceableRandomAccessSlice {
    var baseRange: Range<Base.Index> {
        return startIndex..<endIndex
    }
}

// MARK: - RangeReplaceableCollection
extension Tensor : RangeReplaceableCollection {

    public typealias SubSequence = RangeReplaceableRandomAccessSlice<Tensor<ItemType>>

    public mutating func append(_ newElement: Tensor<ItemType>) {
        precondition(newElement.shape == elementShape, "Element shape mismatch")
        reserveCapacity(count + 1)
        items.append(contentsOf: newElement.items)
    }

    public mutating func reserveCapacity(_ minimumCapacity: Int) {
        let cap = Swift.max(items.capacity, itemCountPerElement * minimumCapacity)
        items.reserveCapacity(cap)
    }

    @discardableResult
    public mutating func remove(at index: Int) -> Tensor<ItemType> {
        precondition(indices.contains(index), "Index out of range")
        let range = itemSubrange(from: index..<index+1)
        defer { items.removeSubrange(range) }
        return Tensor(self[range])
    }

    public mutating func append<S : Sequence>(contentsOf newElements: S)
        where S.Iterator.Element == Tensor<ItemType> {
        for element in newElements {
            append(element)
        }
    }

    public mutating func append<C : Collection>(contentsOf newElements: C)
        where C.Iterator.Element == Tensor<ItemType>, C.IndexDistance == Int {
        reserveCapacity(count + newElements.count)
        for element in newElements {
            append(element)
        }
    }

    public mutating func append(contentsOf newElements: Tensor<ItemType>) {
        precondition(newElements.elementShape == elementShape, "Element shape mismatch")
        reserveCapacity(count + newElements.count)
        items.append(contentsOf: newElements.items)
    }

    public mutating func replaceSubrange<C : Collection>
        (_ subrange: Range<Int>, with newElements: C) where C.Iterator.Element == Tensor<ItemType> {
        let range = itemSubrange(from: subrange)
        let elemShape = elementShape
        items.replaceSubrange(range, with: newElements.lazy.flatMap { elem -> ContiguousArray<ItemType> in
            precondition(elemShape == elem.shape, "Element shape mismatch")
            return elem.items
        })
    }

    public subscript(bounds: Range<Int>) -> RangeReplaceableRandomAccessSlice<Tensor<ItemType>> {
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
            items[itemSubrange(from: bounds)] =
                newValue.base.items[newValue.base.itemSubrange(from: newValue.baseRange)]
        }
    }

}

// MARK: - RandomAccessCollection
extension Tensor : RandomAccessCollection {
    public typealias Index = Int

    /// Access a sub-tensor at index
    public subscript(index: TensorIndex) -> Tensor<ItemType> {
        get {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize as Range
            return Tensor(shape: newShape, items: items[range])
        }
        set {
            precondition(!isScalar || index.isEmpty, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            precondition(newShape == newValue.shape, "Shape mismatch")
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            items.replaceSubrange(range, with: newValue.items)
        }
    }

    /// Access a sub-tensor at an index specified by a list of dimensional indices
    /// - parameter indices: tensor indices
    /// - note: the count of indices must equal the raw rank of the tensor
    public subscript(indices: Int...) -> Tensor<ItemType> {
        get {
            return self[TensorIndex(indices)]
        }
        set {
            self[TensorIndex(indices)] = newValue
        }
    }

    /// Access a sub-tensor at the current dimension at index
    public subscript(index: Int) -> Tensor<ItemType> {
        get {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst()
            let contiguousIndex = itemIndex(fromIndex: index)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            return Tensor(shape: newShape, items: items[range])
        }
        set {
            precondition(!isScalar, "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst()
            let contiguousIndex = itemIndex(fromIndex: index)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            items.replaceSubrange(range, with: newValue.items)
        }
    }

    public var count: Int {
        return isScalar ? 0 : items.count / itemCountPerElement
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

// MARK: - Accessors for tensor slice
public extension RangeReplaceableRandomAccessSlice
    where Base : TensorProtocol, Base.Index == Int, IndexDistance == Int
{
    public var shape: TensorShape {
        return base.elementShape?.prepending(count) ?? .scalar
    }

    public var items: ArraySlice<Base.ItemType> {
        return base.items[base.itemSubrange(from: baseRange)]
    }

    var itemCount: Int {
        return count * base.itemCountPerElement
    }

}

// MARK: - Reshaping
public extension Tensor {

    public func reshaped(as newShape: TensorShape) -> Tensor? {
        guard self.shape.contiguousSize == newShape.contiguousSize else {
            return nil
        }
        return Tensor(shape: newShape, items: items)
    }

}

// MARK: - Unsafe access
public extension Tensor {

    func withUnsafeBufferPointer<Result>
        (_ body: (UnsafeBufferPointer<ItemType>) throws -> Result) rethrows -> Result {
        return try items.withUnsafeBufferPointer { ptr in
            try body(ptr)
        }
    }

    mutating func withUnsafeMutableBufferPointer<Result>
        (_ body: (inout UnsafeMutableBufferPointer<ItemType>) throws -> Result) rethrows -> Result {
        return try items.withUnsafeMutableBufferPointer { ptr in
            try body(&ptr)
        }
    }

}

// MARK: - Textual description
extension Tensor : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        target.write(
            isScalar ? String(describing: items[0])
                     : "[\(map{"\($0)"}.joined(separator: ", "))]"
        )
    }
}
