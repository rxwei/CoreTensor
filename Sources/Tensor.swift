//
//  Tensor.swift
//  DLVM
//
//  Created by Richard Wei on 2/3/17.
//
//

/// Tensor
public struct Tensor<ItemType> {

    /// Sub-tensor (element) shape
    public internal(set) var elementShape: TensorShape? {
        didSet {
            itemCountPerElement = elementShape?.contiguousSize ?? 0
        }
    }

    public var isScalar: Bool {
        return elementShape == nil
    }

    /// Tensor shape
    public var shape: TensorShape {
        return elementShape?.prepending(count) ?? .scalar
    }

    /// The number of items (atoms) per element (sub-tensor).
    /// - Note: This holds the same value as the computed property
    /// `shape.contiguousSize`, which is extremely frequently used.
    /// We cache it in this variable whenever a new shape is set
    public private(set) var itemCountPerElement: Int

    /// Contiguous storage
    fileprivate var _items: ArraySlice<ItemType>

    /// Capacity reserved for sub-tensors
    public var capacity: Int {
        return _items.capacity / itemCountPerElement
    }

    fileprivate init(elementShape: TensorShape?, items: ArraySlice<ItemType>) {
        let elementContiguousSize = elementShape?.contiguousSize ?? 1
        precondition(items.count % elementContiguousSize == 0,
                     "Item count does not match element shape")
        self.itemCountPerElement = elementShape == nil ? 0 : elementContiguousSize
        self.elementShape = elementShape
        self._items = items
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    fileprivate init(shape: TensorShape, items: ArraySlice<ItemType>) {
        precondition(items.count >= shape.contiguousSize,
                     "The slice has fewer elements than required by the shape")
        self.init(elementShape: shape.isScalar ? nil : shape.dropFirst(),
                  items: items.prefix(shape.contiguousSize))
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
    public init<S : Collection>(elementShape: TensorShape, elements: S)
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
        var slice = ArraySlice(items.prefix(contiguousSize))
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
        let items = ArraySlice(repeating: repeatedValue, count: shape.contiguousSize)
        self.init(shape: shape, items: items)
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    public init(shape: TensorShape, supplier: () -> ItemType) {
        let items = ArraySlice((0..<shape.contiguousSize).map { _ in supplier() })
        self.init(shape: shape, items: items)
    }

    internal func itemIndex(fromIndex index: Int) -> Int {
        return itemCountPerElement * index
    }

    internal func itemSubrange(fromSubrange tensorSubrange: Range<Int>) -> Range<Int> {
        return itemIndex(fromIndex: tensorSubrange.lowerBound)
           ..< itemIndex(fromIndex: tensorSubrange.upperBound)
    }
    
}

// MARK: - Element equality and item equality
public extension Tensor where ItemType : Equatable {

    public func elementsEqual(_ other: Tensor<ItemType>) -> Bool {
        return shape.elementsEqual(other.shape) && _items.elementsEqual(other._items)
    }

    public func itemsEqual(_ other: Tensor<ItemType>) -> Bool {
        return _items.elementsEqual(other._items)
    }

}

// MARK: - Isomorphism
public extension Tensor {

    public func isSimilar(to other: Tensor<ItemType>) -> Bool {
        return shape ~ other.shape
    }

    public func isIsomorphic(to other: Tensor<ItemType>) -> Bool {
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
        self.init(elementShape: .scalar, items: ArraySlice(bounds))
    }

    public init(scalarElementsIn bounds: CountableClosedRange<ItemType>) {
        self.init(elementShape: .scalar, items: ArraySlice(bounds))
    }

}

// MARK: - Item mutation
public extension Tensor {

    var itemCount: Int {
        return _items.count
    }

    var items: AnyRandomAccessCollection<ItemType> {
        return AnyRandomAccessCollection(_items)
    }

    func makeItemIterator() -> IndexingIterator<ArraySlice<ItemType>> {
        return _items.makeIterator()
    }

    func item(at index: Int) -> ItemType {
        return _items[_items.startIndex.advanced(by: index)]
    }

    mutating func updateItem(at index: Int, to newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] = newValue
    }

}

public extension Tensor where ItemType : IntegerArithmetic {

    mutating func incrementItem(at index: Int, by newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] += newValue
    }

    mutating func decrementItem(at index: Int, by newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] -= newValue
    }

    mutating func multiplyItem(at index: Int, by newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] *= newValue
    }

    mutating func divideItem(at index: Int, by newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] /= newValue
    }

}

public extension Tensor where ItemType : FloatingPoint {

    mutating func incrementItem(at index: Int, by newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] += newValue
    }

    mutating func decrementItem(at index: Int, by newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] -= newValue
    }

    mutating func multiplyItem(at index: Int, by newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] *= newValue
    }

    mutating func divideItem(at index: Int, by newValue: ItemType) {
        _items[_items.startIndex.advanced(by: index)] /= newValue
    }

}

// MARK: - RangeReplaceableCollection
extension Tensor : RangeReplaceableCollection {

    public mutating func append(_ newElement: Tensor<ItemType>) {
        precondition(newElement.shape == elementShape, "Element shape mismatch")
        reserveCapacity(count + 1)
        _items.append(contentsOf: newElement._items)
    }

    public mutating func reserveCapacity(_ minimumCapacity: Int) {
        let cap = Swift.max(_items.capacity, itemCountPerElement * minimumCapacity)
        _items.reserveCapacity(cap)
    }

    @discardableResult
    public mutating func remove(at index: Int) -> Tensor<ItemType> {
        precondition(indices.contains(index), "Index out of range")
        let range: Range = itemSubrange(fromSubrange: index..<index+1)
        defer { _items.removeSubrange(range) }
        return self[range]
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
        _items.append(contentsOf: newElements._items)
    }

    public mutating func replaceSubrange<C : Collection>
        (_ subrange: Range<Int>, with newElements: C) where C.Iterator.Element == Tensor<ItemType> {
        let storageSubrange = itemSubrange(fromSubrange: subrange)
        _items.replaceSubrange(storageSubrange, with: Tensor(newElements)._items)
    }

    public subscript(bounds: Range<Int>) -> Tensor<ItemType> {
        get {
            guard let elementShape = elementShape else {
                preconditionFailure("I am a scalar and I have no dimensions!")
            }
            return Tensor(shape: elementShape.prepending(bounds.count),
                          items: _items[itemSubrange(fromSubrange: bounds)])
        }
        set {
            guard let elementShape = elementShape else {
                preconditionFailure("I am a scalar and I have no dimensions!")
            }
            precondition(newValue.shape == elementShape, "Element shape mismatch")
            _items[itemSubrange(fromSubrange: bounds)] = newValue._items
        }
    }
    
}

// MARK: - RandomAccessCollection
extension Tensor : RandomAccessCollection {
    public typealias Index = Int
    public typealias SubSequence = Tensor<ItemType>

    /// Access a sub-tensor at index
    public subscript(index: TensorIndex) -> Tensor<ItemType> {
        get {
            precondition(!(isScalar && !index.isEmpty), "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            return Tensor(shape: newShape, items: _items[range])
        }
        set {
            precondition(!(isScalar && !index.isEmpty), "I am a scalar and I have no dimensions!")
            let newShape = shape.dropFirst(index.count)
            let contiguousIndex = index.contiguousIndex(in: shape)
            let range = contiguousIndex..<contiguousIndex+newShape.contiguousSize
            _items[range] = newValue._items
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
            return self[[index]]
        }
        set {
            self[[index]] = newValue
        }
    }

    public var count: Int {
        return isScalar ? 0 : _items.count / itemCountPerElement
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

    /// Returns the index after the specified one in the last dimension
    public func index(after i: TensorIndex) -> TensorIndex {
        guard !i.isEmpty else { return i }
        var newIndex = i
        newIndex[newIndex.endIndex-1] += 1
        return newIndex
    }

    /// Returns the index before the specified one in the last dimension
    public func index(before i: TensorIndex) -> TensorIndex {
        guard !i.isEmpty else { return i }
        var newIndex = i
        newIndex[newIndex.endIndex-1] -= 1
        return newIndex
    }

}

// MARK: - Reshaping
public extension Tensor {

    public func reshaped(as newShape: TensorShape) -> Tensor? {
        guard self.shape.contiguousSize == newShape.contiguousSize else {
            return nil
        }
        return Tensor(shape: newShape, items: _items)
    }
    
}

// MARK: - Unsafe access
public extension Tensor {

    func withUnsafeBufferPointer<Result>
        (_ body: (UnsafeBufferPointer<ItemType>) throws -> Result) rethrows -> Result {
        return try _items.withUnsafeBufferPointer { ptr in
            try body(ptr)
        }
    }

    mutating func withUnsafeMutableBufferPointer<Result>
        (_ body: (inout UnsafeMutableBufferPointer<ItemType>) throws -> Result) rethrows -> Result {
        return try _items.withUnsafeMutableBufferPointer { ptr in
            try body(&ptr)
        }
    }
    
}
