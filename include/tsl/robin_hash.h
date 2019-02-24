/**
 * MIT License
 * 
 * Copyright (c) 2017 Tessil
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef TSL_ROBIN_HASH_H
#define TSL_ROBIN_HASH_H 


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "robin_growth_policy.h"


namespace tsl {
    
namespace detail_robin_hash {

template<typename T>
struct make_void {
    using type = void;
};

template<typename T, typename = void>
struct has_is_transparent: std::false_type {
};

template<typename T>
struct has_is_transparent<T, typename make_void<typename T::is_transparent>::type>: std::true_type {
};

template<typename U>
struct is_power_of_two_policy: std::false_type {
};

template<std::size_t GrowthFactor>
struct is_power_of_two_policy<tsl::rh::power_of_two_growth_policy<GrowthFactor>>: std::true_type {
};


using truncated_hash_type = std::uint_least32_t;

/**
 * Helper class that stores a truncated hash if StoreHash is true and nothing otherwise.
 */
template<bool StoreHash>
class bucket_entry_hash {
public:
    bool bucket_hash_equal(std::size_t /*hash*/) const noexcept {
        return true;
    }
    
    truncated_hash_type truncated_hash() const noexcept {
        return 0;
    }
    
protected:
    void set_hash(truncated_hash_type /*hash*/) noexcept {
    }
};

template<>
class bucket_entry_hash<true> {
public:
    bool bucket_hash_equal(std::size_t hash) const noexcept {
        return m_hash == truncated_hash_type(hash);
    }
    
    truncated_hash_type truncated_hash() const noexcept {
        return m_hash;
    }
    
protected:
    void set_hash(truncated_hash_type hash) noexcept {
        m_hash = truncated_hash_type(hash);
    }
    
private:    
    truncated_hash_type m_hash;
};


/**
 * Each bucket entry has:
 * - A value of type `ValueType`.
 * - An integer to store how far the value of the bucket, if any, is from its ideal bucket 
 *   (ex: if the current bucket 5 has the value 'foo' and `hash('foo') % nb_buckets` == 3,
 *        `dist_from_ideal_bucket()` will return 2 as the current value of the bucket is two
 *        buckets away from its ideal bucket)
 *   If there is no value in the bucket (i.e. `empty()` is true) `dist_from_ideal_bucket()` will be < 0.
 * - A marker which tells us if the bucket is the last bucket of the bucket array (useful for the 
 *   iterator of the hash table).
 * - If `StoreHash` is true, 32 bits of the hash of the value, if any, are also stored in the bucket. 
 *   If the size of the hash is more than 32 bits, it is truncated. We don't store the full hash
 *   as storing the hash is a potential opportunity to use the unused space due to the alignement
 *   of the bucket_entry structure. We can thus potentially store the hash without any extra space 
 *   (which would not be possible with 64 bits of the hash).
 */
template<typename ValueType, bool StoreHash>
class bucket_entry: public bucket_entry_hash<StoreHash> {
    using bucket_hash = bucket_entry_hash<StoreHash>;
    
public:
    using value_type = ValueType;
    using distance_type = std::int_least16_t;
    
    
    bucket_entry() noexcept: bucket_hash(), m_dist_from_ideal_bucket(EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET),
                             m_last_bucket(false)
    {
        tsl_rh_assert(empty());
    }
    
    bucket_entry(bool last_bucket) noexcept: bucket_hash(), m_dist_from_ideal_bucket(EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET),
                                             m_last_bucket(last_bucket)
    {
        tsl_rh_assert(empty());
    }
    
    bucket_entry(const bucket_entry& other) noexcept(std::is_nothrow_copy_constructible<value_type>::value)
            : bucket_hash(other),
              m_dist_from_ideal_bucket(EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET), 
              m_last_bucket(other.m_last_bucket)
    {
        if(!other.empty()) {
            ::new (static_cast<void*>(std::addressof(m_value))) value_type(other.value());
            m_dist_from_ideal_bucket = other.m_dist_from_ideal_bucket;
        }
    }
    
    bucket_entry(bucket_entry&& other) noexcept(std::is_nothrow_move_constructible<value_type>::value)
            : bucket_hash(other),
              m_dist_from_ideal_bucket(EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET),
              m_last_bucket(other.m_last_bucket)
    {
        if(!other.empty()) {
            ::new (static_cast<void*>(std::addressof(m_value))) value_type(std::move(other.value()));
            m_dist_from_ideal_bucket = other.m_dist_from_ideal_bucket;
            other.clear();
        }
    }
    
    bucket_entry& operator=(const bucket_entry& ) = delete;
    bucket_entry& operator=(bucket_entry&& ) = delete;
    
    ~bucket_entry() noexcept {
        clear();
    }
    
    void clear() noexcept {
        if(!empty()) {
            destroy_value();
            m_dist_from_ideal_bucket = EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET;
        }
    }
    
    bool empty() const noexcept {
        return m_dist_from_ideal_bucket == EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET;
    }
    
    value_type& value() noexcept {
        tsl_rh_assert(!empty());
        return *reinterpret_cast<value_type*>(std::addressof(m_value));
    }
    
    const value_type& value() const noexcept {
        tsl_rh_assert(!empty());
        return *reinterpret_cast<const value_type*>(std::addressof(m_value));
    }
    
    distance_type dist_from_ideal_bucket() const noexcept {
        return m_dist_from_ideal_bucket;
    }
    
    bool last_bucket() const noexcept {
        return m_last_bucket;
    }
    
    void set_as_last_bucket() noexcept {
        m_last_bucket = true;
    }
        
    template<typename... Args>
    void set_value_of_empty_bucket(distance_type dist_from_ideal_bucket, 
                                   truncated_hash_type hash, Args&&... value_type_args) 
    {
        tsl_rh_assert(dist_from_ideal_bucket >= 0);
        tsl_rh_assert(empty());
        
        ::new (static_cast<void*>(std::addressof(m_value))) value_type(std::forward<Args>(value_type_args)...);
        this->set_hash(hash);
        m_dist_from_ideal_bucket = dist_from_ideal_bucket;
        
        tsl_rh_assert(!empty());
    }
    
    void swap_with_value_in_bucket(distance_type& dist_from_ideal_bucket, 
                                   truncated_hash_type& hash, value_type& value) 
    {
        tsl_rh_assert(!empty());
        
        using std::swap;
        swap(value, this->value());
        swap(dist_from_ideal_bucket, m_dist_from_ideal_bucket);
        
        // Avoid warning of unused variable if StoreHash is false
        (void) hash;
        if(StoreHash) {
            const truncated_hash_type tmp_hash = this->truncated_hash();
            this->set_hash(hash);
            hash = tmp_hash;
        }
    }
    
    static truncated_hash_type truncate_hash(std::size_t hash) noexcept {
        return truncated_hash_type(hash);
    }
    
private:
    void destroy_value() noexcept {
        tsl_rh_assert(!empty());
        value().~value_type();
    }
    
private:
    using storage = typename std::aligned_storage<sizeof(value_type), alignof(value_type)>::type;
    
    static const distance_type EMPTY_MARKER_DIST_FROM_IDEAL_BUCKET = -1;
    
    distance_type m_dist_from_ideal_bucket;
    bool m_last_bucket;
    storage m_value;
};


/**
 * Sequential allocator-aware container of 'bucket_entry<ValueType, StoreHash>'.
 * The 'operator bucket_entry*()' method always returns a valid pointer to a 'bucket_entry'.
 * When 'buckets' is empty, the returned value is an empty 'bucket_entry'.
 * 
 * We use this class instead of a `std::vector<bucket_entry<ValueType, StoreHash>>` to 
 * reduce the sizeof(robin_hash). With a vector we would need 3*sizeof(void*) for the vector + sizeof(void*) for
 * a pointer that either points to the start of the vector or an empty bucket.
 * The 'bucket' class only requires sizeof(void*) + sizeof(std::size_t).
 */
template<class ValueType,
         class Allocator,
         bool StoreHash>
class buckets: public std::allocator_traits<Allocator>::template rebind_alloc<bucket_entry<ValueType, StoreHash>> {
public:
    using bucket_entry = tsl::detail_robin_hash::bucket_entry<ValueType, StoreHash>;
    using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<bucket_entry>;
    using value_type = ValueType;
    using size_type = std::size_t;
    
public:
    buckets(size_type bucket_count, const Allocator& alloc): allocator_type(alloc),
                                                             m_buckets(static_empty_bucket_ptr()), 
                                                             m_bucket_count(bucket_count)
    {
        if(m_bucket_count > max_size()) {
            TSL_RH_THROW_OR_TERMINATE(std::length_error, "The map exceeds its maxmimum bucket count.");
        }
        
        if(m_bucket_count > 0) {
            m_buckets = std::allocator_traits<allocator_type>::allocate(*this, m_bucket_count);
            uninitialized_default_construct(*this, m_buckets, m_buckets + m_bucket_count);
            
            m_buckets[m_bucket_count - 1].set_as_last_bucket();
        }
    }
    
    ~buckets() {
        deallocate();
    }
    
    buckets(const buckets& other): allocator_type(std::allocator_traits<allocator_type>::select_on_container_copy_construction(other)),
                                   m_buckets(static_empty_bucket_ptr()),
                                   m_bucket_count(other.m_bucket_count)
    {
        if(m_bucket_count > 0) {
            m_buckets = std::allocator_traits<allocator_type>::allocate(*this, m_bucket_count);
            uninitialized_copy(*this, other.m_buckets, other.m_buckets + other.m_bucket_count, m_buckets);
        }
    }
    
    buckets(buckets&& other) noexcept: allocator_type(std::move(static_cast<allocator_type&>(other))),
                                       m_buckets(other.m_buckets),
                                       m_bucket_count(other.m_bucket_count)
    {
        other.m_buckets = static_empty_bucket_ptr();
        other.m_bucket_count = 0;
    }
    
    buckets& operator=(const buckets& other) {
        if(&other != this) {
            if(std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value) {
                if(get_allocator_ref() != other.get_allocator_ref()) {
                    deallocate();
                }
                
                get_allocator_ref() = other.get_allocator_ref();
            }
            
            if(m_bucket_count != other.m_bucket_count) {
                deallocate();
                
                m_buckets = std::allocator_traits<allocator_type>::allocate(*this, other.m_bucket_count);
                m_bucket_count = other.m_bucket_count;
            }
            else {
                destroy(*this, m_buckets, m_buckets + m_bucket_count);
            }
            
            if(m_bucket_count == 0) {
                m_buckets = static_empty_bucket_ptr();
            }
            else {
                uninitialized_copy(*this, other.m_buckets, other.m_buckets + m_bucket_count, m_buckets);
            }
                    
        }
        
        return *this;
    }
    
    buckets& operator=(buckets&& other) {
        if(std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value ||
           get_allocator_ref() == other.get_allocator_ref()) 
        {
            deallocate();
            
            if(std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value) {
                get_allocator_ref() = other.get_allocator_ref();
            }
            
            using std::swap;
            swap(m_buckets, other.m_buckets);
            swap(m_bucket_count, other.m_bucket_count);
        }
        else {
            if(m_bucket_count != other.m_bucket_count) {
                deallocate();
                
                m_buckets = std::allocator_traits<allocator_type>::allocate(*this, other.m_bucket_count);
                m_bucket_count = other.m_bucket_count;
            }
            else {
                destroy(*this, m_buckets, m_buckets + m_bucket_count);
            }
            
            if(m_bucket_count == 0) {
                m_buckets = static_empty_bucket_ptr();
            }
            else {
                uninitialized_move(*this, other.m_buckets, other.m_buckets + m_bucket_count, m_buckets);
                other.deallocate();
            }
        }
        
        return *this;
    }
    
    void swap(buckets& other) {
        using std::swap;
        
        if(std::allocator_traits<allocator_type>::propagate_on_container_swap::value) {
            swap(get_allocator_ref(), other.get_allocator_ref());
        }
        else {
            tsl_rh_assert(get_allocator_ref() == other.get_allocator_ref());
        }
        
        swap(m_buckets, other.m_buckets);
        swap(m_bucket_count, other.m_bucket_count);
    }
    
    void clear() noexcept {
        for(std::size_t i = 0; i < size(); i++) {
            m_buckets[i].clear();
        }
    }

    allocator_type& get_allocator_ref() {
        return static_cast<allocator_type&>(*this);
    }

    const allocator_type& get_allocator_ref() const {
        return static_cast<const allocator_type&>(*this);
    }
    
    size_type size() const {
        return m_bucket_count;
    }
    
    size_type max_size() const noexcept {
        return std::allocator_traits<allocator_type>::max_size(*this);
    }
    
    operator bucket_entry*() noexcept {
        return m_buckets;
    }
    
    operator const bucket_entry*() const noexcept {
        return m_buckets;
    }
    
    friend void swap(buckets& lhs, buckets& rhs) {
        lhs.swap(rhs);
    }

private:
    /**
     * Return an always valid pointer to a static empty bucket_entry with last_bucket() == true.
     */            
    bucket_entry* static_empty_bucket_ptr() {
        static bucket_entry empty_bucket(true);
        return &empty_bucket;
    }
    
    void deallocate() {
        tsl_rh_assert(m_buckets != nullptr);
        if(m_bucket_count == 0) {
            tsl_rh_assert(m_buckets == static_empty_bucket_ptr());
            return;
        }
        
        destroy(*this, m_buckets, m_buckets + m_bucket_count);
        std::allocator_traits<allocator_type>::deallocate(*this, m_buckets, m_bucket_count);
        
        m_buckets = static_empty_bucket_ptr();
        m_bucket_count = 0;
    }
    
    // Equivalent to std::uninitialized_copy with Alloc
    template<class Alloc, class InputIt, class ForwardIt>
    static ForwardIt uninitialized_copy(Alloc& alloc, InputIt first, InputIt last, ForwardIt d_first) {
        ForwardIt current = d_first;
        
        try {
            for (; first != last; ++first, (void) ++current) {
                std::allocator_traits<Alloc>::construct(alloc, std::addressof(*current),  *first);
            }
            
            return current;
        } 
        catch (...) {
            destroy(alloc, d_first, current);
            throw;
        }
    }
    
    // Equivalent to std::uninitialized_move with Alloc
    template<class Alloc, class InputIt, class ForwardIt>
    static ForwardIt uninitialized_move(Alloc& alloc, InputIt first, InputIt last, ForwardIt d_first) {
        ForwardIt current = d_first;
        
        try {
            for (; first != last; ++first, (void) ++current) {
                std::allocator_traits<Alloc>::construct(alloc, std::addressof(*current),  std::move(*first));
            }
            
            return current;
        } 
        catch (...) {
            destroy(alloc, d_first, current);
            throw;
        }
    }

    // Equivalent to std::uninitialized_default_construct with Alloc
    template<class Alloc, class ForwardIt>
    static void uninitialized_default_construct(Alloc& alloc, ForwardIt first, ForwardIt last) {
        ForwardIt current = first;
        
        try {
            for (; current != last; ++current) {
                std::allocator_traits<Alloc>::construct(alloc, std::addressof(*current));
            }
        } 
        catch (...) {
            destroy(alloc, first, current);
            throw;
        }
    }
    
    // Equivalent to std::destroy with Alloc
    template<class Alloc, class ForwardIt>
    static void destroy(Alloc& alloc, ForwardIt first, ForwardIt last) {
        for (; first != last; ++first) {
            std::allocator_traits<Alloc>::destroy(alloc, std::addressof(*first));
        }
    }
    
private:
    bucket_entry* m_buckets;
    size_type m_bucket_count;
};


/**
 * Internal common class used by `robin_map` and `robin_set`. 
 * 
 * ValueType is what will be stored by `robin_hash` (usually `std::pair<Key, T>` for map and `Key` for set).
 * 
 * `KeySelect` should be a `FunctionObject` which takes a `ValueType` in parameter and returns a 
 *  reference to the key.
 * 
 * `ValueSelect` should be a `FunctionObject` which takes a `ValueType` in parameter and returns a 
 *  reference to the value. `ValueSelect` should be void if there is no value (in a set for example).
 * 
 * The strong exception guarantee only holds if the expression 
 * `std::is_nothrow_swappable<ValueType>::value && std::is_nothrow_move_constructible<ValueType>::value` is true.
 * 
 * Behaviour is undefined if the destructor of `ValueType` throws.
 */
template<class ValueType,
         class KeySelect,
         class ValueSelect,
         class Hash,
         class KeyEqual,
         class Allocator,
         bool StoreHash,
         class GrowthPolicy>
class robin_hash: private Hash, private KeyEqual, private GrowthPolicy {
private:    
    template<typename U>
    using has_mapped_type = typename std::integral_constant<bool, !std::is_same<U, void>::value>;
    
    static_assert(noexcept(std::declval<GrowthPolicy>().bucket_for_hash(std::size_t(0))), "GrowthPolicy::bucket_for_hash must be noexcept.");
    static_assert(noexcept(std::declval<GrowthPolicy>().clear()), "GrowthPolicy::clear must be noexcept.");
    
public:
    template<bool IsConst>
    class robin_iterator;
    
    using key_type = typename KeySelect::key_type;
    using value_type = ValueType;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using hasher = Hash;
    using key_equal = KeyEqual;
    using allocator_type = Allocator;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = robin_iterator<false>;
    using const_iterator = robin_iterator<true>;
    
    
private:
    /**
     * Either store the hash because we are asked by the `StoreHash` template parameter
     * or store the hash because it doesn't cost us anything in size and can be used to speed up rehash.
     */
    static constexpr bool STORE_HASH = StoreHash || 
                                       (
                                         (sizeof(tsl::detail_robin_hash::bucket_entry<value_type, true>) ==
                                          sizeof(tsl::detail_robin_hash::bucket_entry<value_type, false>))
                                         &&
                                         (sizeof(std::size_t) == sizeof(truncated_hash_type) ||
                                          is_power_of_two_policy<GrowthPolicy>::value)
                                         &&
                                          // Don't store the hash for primitive types with default hash.
                                          (!std::is_arithmetic<key_type>::value ||
                                           !std::is_same<Hash, std::hash<key_type>>::value)
                                       );
                                        
    /**
     * Only use the stored hash on lookup if we are explictly asked. We are not sure how slow
     * the KeyEqual operation is. An extra comparison may slow things down with a fast KeyEqual.
     */
    static constexpr bool USE_STORED_HASH_ON_LOOKUP = StoreHash;

    /**
     * We can only use the hash on rehash if the size of the hash type is the same as the stored one or
     * if we use a power of two modulo. In the case of the power of two modulo, we just mask
     * the least significant bytes, we just have to check that the truncated_hash_type didn't truncated
     * more bytes.
     */
    static bool USE_STORED_HASH_ON_REHASH(size_type bucket_count) {
        (void) bucket_count;
        if(STORE_HASH && sizeof(std::size_t) == sizeof(truncated_hash_type)) {
            return true;
        }
        else if(STORE_HASH && is_power_of_two_policy<GrowthPolicy>::value) {
            tsl_rh_assert(bucket_count > 0);
            return (bucket_count - 1) <= std::numeric_limits<truncated_hash_type>::max();
        }
        else {
            return false;   
        }
    }
    
private:
    using buckets = tsl::detail_robin_hash::buckets<ValueType, Allocator, STORE_HASH>;
    using bucket_entry = typename buckets::bucket_entry;
    using distance_type = typename bucket_entry::distance_type;
    
public: 
    /**
     * The 'operator*()' and 'operator->()' methods return a const reference and const pointer respectively to the 
     * stored value type.
     * 
     * In case of a map, to get a mutable reference to the value associated to a key (the '.second' in the 
     * stored pair), you have to call 'value()'. 
     * 
     * The main reason for this is that if we returned a `std::pair<Key, T>&` instead 
     * of a `const std::pair<Key, T>&`, the user may modify the key which will put the map in a undefined state.
     */
    template<bool IsConst>
    class robin_iterator {
        friend class robin_hash;
        
    private:
        using bucket_entry_ptr = typename std::conditional<IsConst, 
                                                           const bucket_entry*, 
                                                           bucket_entry*>::type;
    
        
        robin_iterator(bucket_entry_ptr bucket) noexcept: m_bucket(bucket) {
        }
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const typename robin_hash::value_type;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using pointer = value_type*;
        
        
        robin_iterator() noexcept {
        }
        
        // Copy constructor from iterator to const_iterator.
        template<bool TIsConst = IsConst, typename std::enable_if<TIsConst>::type* = nullptr>
        robin_iterator(const robin_iterator<!TIsConst>& other) noexcept: m_bucket(other.m_bucket) {
        }
        
        robin_iterator(const robin_iterator& other) = default;
        robin_iterator(robin_iterator&& other) = default;
        robin_iterator& operator=(const robin_iterator& other) = default;
        robin_iterator& operator=(robin_iterator&& other) = default;
        
        const typename robin_hash::key_type& key() const {
            return KeySelect()(m_bucket->value());
        }

        template<class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value && IsConst>::type* = nullptr>
        const typename U::value_type& value() const {
            return U()(m_bucket->value());
        }

        template<class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value && !IsConst>::type* = nullptr>
        typename U::value_type& value() {
            return U()(m_bucket->value());
        }
        
        reference operator*() const {
            return m_bucket->value();
        }
        
        pointer operator->() const {
            return std::addressof(m_bucket->value());
        }
        
        robin_iterator& operator++() {
            while(true) {
                if(m_bucket->last_bucket()) {
                    ++m_bucket;
                    return *this;
                }
                
                ++m_bucket;
                if(!m_bucket->empty()) {
                    return *this;
                }
            }
        }
        
        robin_iterator operator++(int) {
            robin_iterator tmp(*this);
            ++*this;
            
            return tmp;
        }
        
        friend bool operator==(const robin_iterator& lhs, const robin_iterator& rhs) { 
            return lhs.m_bucket == rhs.m_bucket; 
        }
        
        friend bool operator!=(const robin_iterator& lhs, const robin_iterator& rhs) { 
            return !(lhs == rhs); 
        }
        
    private:
        bucket_entry_ptr m_bucket;
    };

    
public:
    robin_hash(size_type bucket_count, 
               const Hash& hash,
               const KeyEqual& equal,
               const Allocator& alloc,
               float max_load_factor): Hash(hash), 
                                       KeyEqual(equal),
                                       GrowthPolicy(bucket_count),
                                       m_buckets(bucket_count, alloc),
                                       m_nb_elements(0), 
                                       m_grow_on_next_insert(false)
    {
        this->max_load_factor(max_load_factor);
    }
    
    robin_hash(const robin_hash& other) = default;
    
    robin_hash(robin_hash&& other) noexcept(std::is_nothrow_move_constructible<Hash>::value &&
                                            std::is_nothrow_move_constructible<KeyEqual>::value &&
                                            std::is_nothrow_move_constructible<GrowthPolicy>::value &&
                                            std::is_nothrow_move_constructible<buckets>::value)
                                          : Hash(std::move(static_cast<Hash&>(other))),
                                            KeyEqual(std::move(static_cast<KeyEqual&>(other))),
                                            GrowthPolicy(std::move(static_cast<GrowthPolicy&>(other))),
                                            m_buckets(std::move(other.m_buckets)),
                                            m_nb_elements(other.m_nb_elements),
                                            m_load_threshold(other.m_load_threshold),
                                            m_max_load_factor(other.m_max_load_factor),
                                            m_grow_on_next_insert(other.m_grow_on_next_insert)
    {
        other.GrowthPolicy::clear();
        other.m_nb_elements = 0;
        other.m_load_threshold = 0;
        other.m_grow_on_next_insert = false;
    }
    
    robin_hash& operator=(const robin_hash& other) = default;
    
    robin_hash& operator=(robin_hash&& other) {
        static_cast<Hash&>(*this) = std::move(static_cast<Hash&>(other));
        static_cast<KeyEqual&>(*this) = std::move(static_cast<KeyEqual&>(other));
        static_cast<GrowthPolicy&>(*this) = std::move(static_cast<GrowthPolicy&>(other));
        m_buckets = std::move(other.m_buckets);
        m_nb_elements = other.m_nb_elements;
        m_load_threshold = other.m_load_threshold;
        m_max_load_factor = other.m_max_load_factor;
        m_grow_on_next_insert = other.m_grow_on_next_insert;
        
        other.GrowthPolicy::clear();
        other.clear();
        
        return *this;
    }
    
    void swap(robin_hash& other) {
        using std::swap;
        
        swap(static_cast<Hash&>(*this), static_cast<Hash&>(other));
        swap(static_cast<KeyEqual&>(*this), static_cast<KeyEqual&>(other));
        swap(static_cast<GrowthPolicy&>(*this), static_cast<GrowthPolicy&>(other));
        swap(m_buckets, other.m_buckets);
        swap(m_nb_elements, other.m_nb_elements);
        swap(m_load_threshold, other.m_load_threshold);
        swap(m_max_load_factor, other.m_max_load_factor);
        swap(m_grow_on_next_insert, other.m_grow_on_next_insert);
    }
    
    
    allocator_type get_allocator() const {
        return m_buckets.get_allocator_ref();
    }
    
    
    /*
     * Iterators
     */
    iterator begin() noexcept {
        std::size_t i = 0;
        while(i < m_buckets.size() && m_buckets[i].empty()) {
            i++;
        }
        
        return iterator(m_buckets + i);
    }
    
    const_iterator begin() const noexcept {
        return cbegin();
    }
    
    const_iterator cbegin() const noexcept {
        std::size_t i = 0;
        while(i < m_buckets.size() && m_buckets[i].empty()) {
            i++;
        }
        
        return const_iterator(m_buckets + i);
    }
    
    iterator end() noexcept {
        return iterator(m_buckets + m_buckets.size());
    }
    
    const_iterator end() const noexcept {
        return cend();
    }
    
    const_iterator cend() const noexcept {
        return const_iterator(m_buckets + m_buckets.size());
    }
    
    
    /*
     * Capacity
     */
    bool empty() const noexcept {
        return m_nb_elements == 0;
    }
    
    size_type size() const noexcept {
        return m_nb_elements;
    }
    
    size_type max_size() const noexcept {
        return m_buckets.max_size();
    }
    
    /*
     * Modifiers
     */
    void clear() noexcept {
        m_buckets.clear();
        m_nb_elements = 0;
        m_load_threshold = 0;
        m_grow_on_next_insert = false;
    }
    
    
    
    template<typename P>
    std::pair<iterator, bool> insert(P&& value) {
        return insert_impl(KeySelect()(value), std::forward<P>(value));
    }
    
    template<typename P>
    iterator insert_hint(const_iterator hint, P&& value) { 
        if(hint != cend() && compare_keys(KeySelect()(*hint), KeySelect()(value))) { 
            return mutable_iterator(hint); 
        }
        
        return insert(std::forward<P>(value)).first; 
    }
    
    template<class InputIt>
    void insert(InputIt first, InputIt last) {
        if(std::is_base_of<std::forward_iterator_tag, 
                           typename std::iterator_traits<InputIt>::iterator_category>::value) 
        {
            const auto nb_elements_insert = std::distance(first, last);
            const size_type nb_free_buckets = m_load_threshold - size();
            tsl_rh_assert(m_load_threshold >= size());
            
            if(nb_elements_insert > 0 && nb_free_buckets < size_type(nb_elements_insert)) {
                reserve(size() + size_type(nb_elements_insert));
            }
        }
        
        for(; first != last; ++first) {
            insert(*first);
        }
    }
    
    
    
    template<class K, class M>
    std::pair<iterator, bool> insert_or_assign(K&& key, M&& obj) { 
        auto it = try_emplace(std::forward<K>(key), std::forward<M>(obj));
        if(!it.second) {
            it.first.value() = std::forward<M>(obj);
        }
        
        return it;
    }
    
    template<class K, class M>
    iterator insert_or_assign(const_iterator hint, K&& key, M&& obj) {
        if(hint != cend() && compare_keys(KeySelect()(*hint), key)) { 
            auto it = mutable_iterator(hint); 
            it.value() = std::forward<M>(obj);
            
            return it;
        }
        
        return insert_or_assign(std::forward<K>(key), std::forward<M>(obj)).first;
    }

    
    template<class... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return insert(value_type(std::forward<Args>(args)...));
    }
    
    template<class... Args>
    iterator emplace_hint(const_iterator hint, Args&&... args) {
        return insert_hint(hint, value_type(std::forward<Args>(args)...));        
    }
    
    
    
    template<class K, class... Args>
    std::pair<iterator, bool> try_emplace(K&& key, Args&&... args) {
        return insert_impl(key, std::piecewise_construct, 
                                std::forward_as_tuple(std::forward<K>(key)), 
                                std::forward_as_tuple(std::forward<Args>(args)...));
    }
    
    template<class K, class... Args>
    iterator try_emplace_hint(const_iterator hint, K&& key, Args&&... args) { 
        if(hint != cend() && compare_keys(KeySelect()(*hint), key)) { 
            return mutable_iterator(hint); 
        }
        
        return try_emplace(std::forward<K>(key), std::forward<Args>(args)...).first;
    }
    
    /**
     * Here to avoid `template<class K> size_type erase(const K& key)` being used when
     * we use an `iterator` instead of a `const_iterator`.
     */
    iterator erase(iterator pos) {
        erase_from_bucket(pos);
        
        /**
         * Erase bucket used a backward shift after clearing the bucket.
         * Check if there is a new value in the bucket, if not get the next non-empty.
         */
        if(pos.m_bucket->empty()) {
            ++pos;
        }
        
        return pos;
    }
    
    iterator erase(const_iterator pos) {
        return erase(mutable_iterator(pos));
    }
    
    iterator erase(const_iterator first, const_iterator last) {
        if(first == last) {
            return mutable_iterator(first);
        }
        
        auto first_mutable = mutable_iterator(first);
        auto last_mutable = mutable_iterator(last);
        for(auto it = first_mutable.m_bucket; it != last_mutable.m_bucket; ++it) {
            if(!it->empty()) {
                it->clear();
                m_nb_elements--;
            }
        }
        
        if(last_mutable == end()) {
            return end();
        }
        
        
        /*
         * Backward shift on the values which come after the deleted values.
         * We try to move the values closer to their ideal bucket.
         */
        std::size_t icloser_bucket = static_cast<std::size_t>(first_mutable.m_bucket - m_buckets);
        std::size_t ito_move_closer_value = static_cast<std::size_t>(last_mutable.m_bucket - m_buckets);
        tsl_rh_assert(ito_move_closer_value > icloser_bucket);
        
        const std::size_t ireturn_bucket = ito_move_closer_value - 
                                           std::min(ito_move_closer_value - icloser_bucket, 
                                                    std::size_t(m_buckets[ito_move_closer_value].dist_from_ideal_bucket()));
        
        while(ito_move_closer_value < m_buckets.size() && m_buckets[ito_move_closer_value].dist_from_ideal_bucket() > 0) {
            icloser_bucket = ito_move_closer_value - 
                             std::min(ito_move_closer_value - icloser_bucket, 
                                      std::size_t(m_buckets[ito_move_closer_value].dist_from_ideal_bucket()));
            
            
            tsl_rh_assert(m_buckets[icloser_bucket].empty());
            const distance_type new_distance = distance_type(m_buckets[ito_move_closer_value].dist_from_ideal_bucket() -
                                                             (ito_move_closer_value - icloser_bucket));
            m_buckets[icloser_bucket].set_value_of_empty_bucket(new_distance, 
                                                                m_buckets[ito_move_closer_value].truncated_hash(), 
                                                                std::move(m_buckets[ito_move_closer_value].value()));
            m_buckets[ito_move_closer_value].clear();
            
            
            ++icloser_bucket;
            ++ito_move_closer_value;
        }

        
        return iterator(m_buckets + ireturn_bucket);
    }
    
    
    template<class K>
    size_type erase(const K& key) {
        return erase(key, hash_key(key));
    }
    
    template<class K>
    size_type erase(const K& key, std::size_t hash) {
        auto it = find(key, hash);
        if(it != end()) {
            erase_from_bucket(it);
            
            return 1;
        }
        else {
            return 0;
        }
    }
    
    
    /*
     * Lookup
     */
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    typename U::value_type& at(const K& key) {
        return at(key, hash_key(key));
    }
    
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    typename U::value_type& at(const K& key, std::size_t hash) {
        return const_cast<typename U::value_type&>(static_cast<const robin_hash*>(this)->at(key, hash));
    }
    
    
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    const typename U::value_type& at(const K& key) const {
        return at(key, hash_key(key));
    }
    
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    const typename U::value_type& at(const K& key, std::size_t hash) const {
        auto it = find(key, hash);
        if(it != cend()) {
            return it.value();
        }
        else {
            TSL_RH_THROW_OR_TERMINATE(std::out_of_range, "Couldn't find key.");
        }
    }
    
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    typename U::value_type& operator[](K&& key) {
        return try_emplace(std::forward<K>(key)).first.value();
    }
    
    
    template<class K>
    size_type count(const K& key) const {
        return count(key, hash_key(key));
    }
    
    template<class K>
    size_type count(const K& key, std::size_t hash) const {
        if(find(key, hash) != cend()) {
            return 1;
        }
        else {
            return 0;
        }
    }
    
    
    template<class K>
    iterator find(const K& key) {
        return find_impl(key, hash_key(key));
    }
    
    template<class K>
    iterator find(const K& key, std::size_t hash) {
        return find_impl(key, hash);
    }
    
    
    template<class K>
    const_iterator find(const K& key) const {
        return find_impl(key, hash_key(key));
    }
    
    template<class K>
    const_iterator find(const K& key, std::size_t hash) const {
        return find_impl(key, hash);
    }
    
    
    template<class K>
    std::pair<iterator, iterator> equal_range(const K& key) {
        return equal_range(key, hash_key(key));
    }
    
    template<class K>
    std::pair<iterator, iterator> equal_range(const K& key, std::size_t hash) {
        iterator it = find(key, hash);
        return std::make_pair(it, (it == end())?it:std::next(it));
    }
    
    
    template<class K>
    std::pair<const_iterator, const_iterator> equal_range(const K& key) const {
        return equal_range(key, hash_key(key));
    }
    
    template<class K>
    std::pair<const_iterator, const_iterator> equal_range(const K& key, std::size_t hash) const {
        const_iterator it = find(key, hash);
        return std::make_pair(it, (it == cend())?it:std::next(it));
    }
    
    /*
     * Bucket interface 
     */
    size_type bucket_count() const {
        return m_buckets.size(); 
    }
    
    size_type max_bucket_count() const {
        return std::min(GrowthPolicy::max_bucket_count(), m_buckets.max_size());
    }
    
    /*
     * Hash policy 
     */
    float load_factor() const {
        if(bucket_count() == 0) {
            return 0;
        }
        
        return float(m_nb_elements)/float(bucket_count());
    }
    
    float max_load_factor() const {
        return m_max_load_factor;
    }
    
    void max_load_factor(float ml) {
        m_max_load_factor = std::max(0.1f, std::min(ml, 0.95f));
        m_load_threshold = size_type(float(bucket_count())*m_max_load_factor);
    }
    
    void rehash(size_type count) {
        count = std::max(count, size_type(std::ceil(float(size())/max_load_factor())));
        rehash_impl(count);
    }
    
    void reserve(size_type count) {
        rehash(size_type(std::ceil(float(count)/max_load_factor())));
    }    
    
    /*
     * Observers
     */
    hasher hash_function() const {
        return static_cast<const Hash&>(*this);
    }
    
    key_equal key_eq() const {
        return static_cast<const KeyEqual&>(*this);
    }
    
    
    /*
     * Other
     */    
    iterator mutable_iterator(const_iterator pos) {
        return iterator(const_cast<bucket_entry*>(pos.m_bucket));
    }
    
private:
    template<class K>
    std::size_t hash_key(const K& key) const {
        return Hash::operator()(key);
    }
    
    template<class K1, class K2>
    bool compare_keys(const K1& key1, const K2& key2) const {
        return KeyEqual::operator()(key1, key2);
    }
    
    std::size_t bucket_for_hash(std::size_t hash) const {
        const std::size_t bucket = GrowthPolicy::bucket_for_hash(hash);
        tsl_rh_assert(bucket < bucket_count() || (bucket == 0 && bucket_count() == 0));
        
        return bucket;
    }
    
    template<class U = GrowthPolicy, typename std::enable_if<is_power_of_two_policy<U>::value>::type* = nullptr>
    std::size_t next_bucket(std::size_t index) const noexcept {
        tsl_rh_assert(index < bucket_count());
        
        return (index + 1) & this->m_mask;
    }
    
    template<class U = GrowthPolicy, typename std::enable_if<!is_power_of_two_policy<U>::value>::type* = nullptr>
    std::size_t next_bucket(std::size_t index) const noexcept {
        tsl_rh_assert(index < bucket_count());
        
        index++;
        return (index != bucket_count())?index:0;
    }
    
    
    
    template<class K>
    iterator find_impl(const K& key, std::size_t hash) {
        return mutable_iterator(static_cast<const robin_hash*>(this)->find(key, hash));
    }
    
    template<class K>
    const_iterator find_impl(const K& key, std::size_t hash) const {
        std::size_t ibucket = bucket_for_hash(hash); 
        distance_type dist_from_ideal_bucket = 0;
        
        while(dist_from_ideal_bucket <= m_buckets[ibucket].dist_from_ideal_bucket()) {
            if(TSL_RH_LIKELY((!USE_STORED_HASH_ON_LOOKUP || m_buckets[ibucket].bucket_hash_equal(hash)) && 
               compare_keys(KeySelect()(m_buckets[ibucket].value()), key))) 
            {
                return const_iterator(m_buckets + ibucket);
            }
            
            ibucket = next_bucket(ibucket);
            dist_from_ideal_bucket++;
        }
        
        return cend();
    }
    
    void erase_from_bucket(iterator pos) {
        pos.m_bucket->clear();
        m_nb_elements--;
        
        /**
         * Backward shift, swap the empty bucket, previous_ibucket, with the values on its right, ibucket,
         * until we cross another empty bucket or if the other bucket has a distance_from_ideal_bucket == 0.
         * 
         * We try to move the values closer to their ideal bucket.
         */
        std::size_t previous_ibucket = static_cast<std::size_t>(pos.m_bucket - m_buckets);
        std::size_t ibucket = next_bucket(previous_ibucket);
        
        while(m_buckets[ibucket].dist_from_ideal_bucket() > 0) {
            tsl_rh_assert(m_buckets[previous_ibucket].empty());
            
            const distance_type new_distance = distance_type(m_buckets[ibucket].dist_from_ideal_bucket() - 1);
            m_buckets[previous_ibucket].set_value_of_empty_bucket(new_distance, m_buckets[ibucket].truncated_hash(), 
                                                                  std::move(m_buckets[ibucket].value()));
            m_buckets[ibucket].clear();

            previous_ibucket = ibucket;
            ibucket = next_bucket(ibucket);
        }
    }
    
    template<class K, class... Args>
    std::pair<iterator, bool> insert_impl(const K& key, Args&&... value_type_args) {
        const std::size_t hash = hash_key(key);
        
        std::size_t ibucket = bucket_for_hash(hash); 
        distance_type dist_from_ideal_bucket = 0;
        
        while(dist_from_ideal_bucket <= m_buckets[ibucket].dist_from_ideal_bucket()) {
            if((!USE_STORED_HASH_ON_LOOKUP || m_buckets[ibucket].bucket_hash_equal(hash)) &&
               compare_keys(KeySelect()(m_buckets[ibucket].value()), key)) 
            {
                return std::make_pair(iterator(m_buckets + ibucket), false);
            }
            
            ibucket = next_bucket(ibucket);
            dist_from_ideal_bucket++;
        }
        
        if(grow_on_high_load()) {
            ibucket = bucket_for_hash(hash);
            dist_from_ideal_bucket = 0;
            
            while(dist_from_ideal_bucket <= m_buckets[ibucket].dist_from_ideal_bucket()) {
                ibucket = next_bucket(ibucket);
                dist_from_ideal_bucket++;
            }
        }
 
        
        if(m_buckets[ibucket].empty()) {
            m_buckets[ibucket].set_value_of_empty_bucket(dist_from_ideal_bucket, bucket_entry::truncate_hash(hash),
                                                         std::forward<Args>(value_type_args)...);
        }
        else {
            insert_value(ibucket, dist_from_ideal_bucket, bucket_entry::truncate_hash(hash), 
                         std::forward<Args>(value_type_args)...);
        }
        
        
        m_nb_elements++;
        /*
         * The value will be inserted in ibucket in any case, either because it was
         * empty or by stealing the bucket (robin hood). 
         */
        return std::make_pair(iterator(m_buckets + ibucket), true);
    }
    
    
    template<class... Args>
    void insert_value(std::size_t ibucket, distance_type dist_from_ideal_bucket, 
                      truncated_hash_type hash, Args&&... value_type_args) 
    {
        value_type value(std::forward<Args>(value_type_args)...);
        insert_value_impl(ibucket, dist_from_ideal_bucket, hash, value);
    }

    void insert_value(std::size_t ibucket, distance_type dist_from_ideal_bucket,
                      truncated_hash_type hash, value_type&& value)
    {
        insert_value_impl(ibucket, dist_from_ideal_bucket, hash, value);
    }

    /*
     * We don't use `value_type&& value` as last argument due to a bug in MSVC when `value_type` is a pointer,
     * The compiler is not able to see the difference between `std::string*` and `std::string*&&` resulting in 
     * compile error.
     * 
     * The `value` will be in a moved state at the end of the function.
     */
    void insert_value_impl(std::size_t ibucket, distance_type dist_from_ideal_bucket,
                           truncated_hash_type hash, value_type& value)
    {
        m_buckets[ibucket].swap_with_value_in_bucket(dist_from_ideal_bucket, hash, value);
        ibucket = next_bucket(ibucket);
        dist_from_ideal_bucket++;
        
        while(!m_buckets[ibucket].empty()) {
            if(dist_from_ideal_bucket > m_buckets[ibucket].dist_from_ideal_bucket()) {
                if(dist_from_ideal_bucket >= REHASH_ON_HIGH_NB_PROBES__NPROBES && 
                   load_factor() >= REHASH_ON_HIGH_NB_PROBES__MIN_LOAD_FACTOR) 
                {
                    /**
                     * The number of probes is really high, rehash the map on the next insert.
                     * Difficult to do now as rehash may throw an exception.
                     */
                    m_grow_on_next_insert = true;
                }
            
                m_buckets[ibucket].swap_with_value_in_bucket(dist_from_ideal_bucket, hash, value);
            }
            
            ibucket = next_bucket(ibucket);
            dist_from_ideal_bucket++;
        }
        
        m_buckets[ibucket].set_value_of_empty_bucket(dist_from_ideal_bucket, hash, std::move(value));
    }
    
    
    void rehash_impl(size_type count) {
        robin_hash new_table(count, static_cast<Hash&>(*this), static_cast<KeyEqual&>(*this), 
                             m_buckets.get_allocator_ref(), m_max_load_factor);
        
        const bool use_stored_hash = USE_STORED_HASH_ON_REHASH(new_table.bucket_count());
        for(std::size_t ibucket = 0; ibucket < m_buckets.size(); ibucket++) {
            if(m_buckets[ibucket].empty()) { 
                continue; 
            }
            
            const std::size_t hash = use_stored_hash?m_buckets[ibucket].truncated_hash():
                                                     new_table.hash_key(KeySelect()(m_buckets[ibucket].value()));
                                                     
            new_table.insert_value_on_rehash(new_table.bucket_for_hash(hash), 0, 
                                             bucket_entry::truncate_hash(hash), std::move(m_buckets[ibucket].value()));
        }
        
        new_table.m_nb_elements = m_nb_elements;
        new_table.swap(*this);
    }
    
    void insert_value_on_rehash(std::size_t ibucket, distance_type dist_from_ideal_bucket, 
                                truncated_hash_type hash, value_type&& value) 
    {
        while(true) {
            if(dist_from_ideal_bucket > m_buckets[ibucket].dist_from_ideal_bucket()) {
                if(m_buckets[ibucket].empty()) {
                    m_buckets[ibucket].set_value_of_empty_bucket(dist_from_ideal_bucket, hash, std::move(value));
                    return;
                }
                else {
                    m_buckets[ibucket].swap_with_value_in_bucket(dist_from_ideal_bucket, hash, value);
                }
            }
            
            dist_from_ideal_bucket++;
            ibucket = next_bucket(ibucket);
        }
    }
    
    
    
    /**
     * Return true if the map has been rehashed.
     */
    bool grow_on_high_load() {
        if(m_grow_on_next_insert || size() >= m_load_threshold) {
            rehash_impl(GrowthPolicy::next_bucket_count());
            m_grow_on_next_insert = false;
            
            return true;
        }
        
        return false;
    }

    
public:
    static const size_type DEFAULT_INIT_BUCKETS_SIZE = 0;
    static constexpr float DEFAULT_MAX_LOAD_FACTOR = 0.5f;
    
private:
    static const distance_type REHASH_ON_HIGH_NB_PROBES__NPROBES = 128;
    static constexpr float REHASH_ON_HIGH_NB_PROBES__MIN_LOAD_FACTOR = 0.15f;
    
private:
    buckets m_buckets;
    size_type m_nb_elements;
    
    size_type m_load_threshold;
    float m_max_load_factor;
    
    bool m_grow_on_next_insert;
};

}

}

#endif
