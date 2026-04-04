use super::*;
use search_tree::*;
use atomics::*;

/// # Safety
/// Implementations must follow the contract on `insert`: if a value is inserted,
/// `None` must be returned. Violating this leads to memory unsafety (double-free).
pub unsafe trait TranspositionTable<Spec: MCTS>: Sync + Sized {
    /// **If this function inserts a value, it must return `None`.** Failure to follow
    /// this rule will lead to memory safety violation.
    ///
    /// Attempts to insert a key/value pair.
    ///
    /// If the key is not present, the table *may* insert it. If the table does
    /// not insert it, the table may either return `None` or a reference to another
    /// value existing in the table. (The latter is allowed so that the table doesn't
    /// necessarily need to handle hash collisions, but it will negatively affect the accuracy
    /// of the search.)
    ///
    /// If the key is present, the table may either:
    /// - Leave the table unchanged and return `Some(reference to associated value)`.
    /// - Leave the table unchanged and return `None`.
    ///
    /// The table *may* choose to replace old values.
    /// The table is *not* responsible for dropping values that are replaced.
    fn insert<'a>(&'a self, key: &Spec::State, value: &'a SearchNode<Spec>,
            handle: SearchHandle<Spec>) -> Option<&'a SearchNode<Spec>>;

    /// Looks up a key.
    ///
    /// If the key is not present, the table *should almost always* return `None`.
    ///
    /// If the key is present, the table *may return either* `None` or a reference
    /// to the associated value.
    fn lookup<'a>(&'a self, key: &Spec::State, handle: SearchHandle<Spec>)
            -> Option<&'a SearchNode<Spec>>;

    /// Clear all entries from the table.
    /// Called during tree re-rooting to prevent dangling pointers.
    fn clear(&mut self) {}
}

unsafe impl<Spec: MCTS<TranspositionTable=Self>> TranspositionTable<Spec> for () {
    fn insert<'a>(&'a self, _: &Spec::State, _: &'a SearchNode<Spec>,
            _: SearchHandle<Spec>) -> Option<&'a SearchNode<Spec>> {
        None
    }

    fn lookup<'a>(&'a self, _: &Spec::State, _: SearchHandle<Spec>)
            -> Option<&'a SearchNode<Spec>> {
        None
    }
}

/// Trait for game states that can be hashed for transposition table lookup.
///
/// The hash must be consistent: equal states must produce equal hashes.
/// Hash `0` is reserved and will not be inserted into the table.
pub trait TranspositionHash {
    /// Compute a hash of this game state. Must return nonzero for insertable states.
    fn hash(&self) -> u64;
}

/// A lock-free hash table using quadratic probing with approximate semantics.
///
/// Hash collisions may cause states to share tree nodes (trading accuracy
/// for memory efficiency). The table does not handle hash collisions precisely —
/// this is by design for MCTS where approximate sharing is acceptable.
pub struct ApproxQuadraticProbingHashTable<K: TranspositionHash, V> {
    arr: Box<[Entry16<K, V>]>,
    capacity: usize,
    mask: usize,
    size: AtomicUsize,
}

struct Entry16<K: TranspositionHash, V> {
    k: AtomicU64,
    v: AtomicPtr<V>,
    _marker: std::marker::PhantomData<K>,
}

impl<K: TranspositionHash, V> Default for Entry16<K, V> {
    fn default() -> Self {
        Self {
            k: Default::default(),
            v: Default::default(),
            _marker: Default::default(),
        }
    }
}
impl<K: TranspositionHash, V> Clone for Entry16<K, V> {
    fn clone(&self) -> Self {
        Self {
            k: AtomicU64::new(self.k.load(Ordering::Relaxed)),
            v: AtomicPtr::new(self.v.load(Ordering::Relaxed)),
            _marker: Default::default(),
        }
    }
}

impl<K: TranspositionHash, V> ApproxQuadraticProbingHashTable<K, V> {
    /// Create a table with the given capacity (must be a power of 2).
    pub fn new(capacity: usize) -> Self {
        assert!(std::mem::size_of::<Entry16<K, V>>() <= 16);
        assert!(capacity.count_ones() == 1, "the capacity must be a power of 2");
        let arr = vec![Entry16::default(); capacity].into_boxed_slice();
        let mask = capacity - 1;
        Self {arr, mask, capacity, size: AtomicUsize::default()}
    }
    /// Create a table large enough to hold `num` entries with room to spare.
    pub fn enough_to_hold(num: usize) -> Self {
        let mut capacity = 1;
        while capacity * 2 < num * 3 {
            capacity <<= 1;
        }
        Self::new(capacity)
    }
}

unsafe impl<K: TranspositionHash, V> Sync for ApproxQuadraticProbingHashTable<K, V> {}
unsafe impl<K: TranspositionHash, V> Send for ApproxQuadraticProbingHashTable<K, V> {}

/// Convenience alias for an approximate transposition table keyed by game state.
pub type ApproxTable<Spec> =
         ApproxQuadraticProbingHashTable<<Spec as MCTS>::State, SearchNode<Spec>>;

fn get_or_write<'a, V>(ptr: &AtomicPtr<V>, v: &'a V) -> Option<&'a V> {
    let result = ptr.compare_exchange(
        std::ptr::null_mut(),
        v as *const _ as *mut _,
        Ordering::Relaxed,
        Ordering::Relaxed);
    convert(result.unwrap_or_else(|x| x))
}

fn convert<'a, V>(ptr: *const V) -> Option<&'a V> {
    if ptr.is_null() {
        None
    } else {
        unsafe { Some(&*ptr) }
    }
}

const PROBE_LIMIT: usize = 16;

unsafe impl<Spec> TranspositionTable<Spec> for ApproxTable<Spec>
    where Spec::State: TranspositionHash, Spec: MCTS
{
    fn insert<'a>(&'a self, key: &Spec::State, value: &'a SearchNode<Spec>,
            handle: SearchHandle<Spec>) -> Option<&'a SearchNode<Spec>> {
        if self.size.load(Ordering::Relaxed) * 3 > self.capacity * 2 {
            return self.lookup(key, handle);
        }
        let my_hash = key.hash();
        if my_hash == 0 {
            return None;
        }
        let mut posn = my_hash as usize & self.mask;
        for inc in 1..(PROBE_LIMIT + 1) {
            let entry = unsafe { self.arr.get_unchecked(posn) };
            let key_here = entry.k.load(Ordering::Relaxed);
            if key_here == my_hash {
                let value_here = entry.v.load(Ordering::Relaxed);
                if !value_here.is_null() {
                    return unsafe { Some(&*value_here) };
                }
                return get_or_write(&entry.v, value);
            }
            if key_here == 0 {
                let key_here = entry.k.compare_exchange(0, my_hash, Ordering::Relaxed, Ordering::Relaxed)
                    .unwrap_or_else(|x| x);
                self.size.fetch_add(1, Ordering::Relaxed);
                if key_here == 0 || key_here == my_hash {
                    return get_or_write(&entry.v, value);
                }
            }
            posn += inc;
            posn &= self.mask;
        }
        None
    }
    fn clear(&mut self) {
        for entry in self.arr.iter_mut() {
            *entry.k.get_mut() = 0;
            *entry.v.get_mut() = std::ptr::null_mut();
        }
        *self.size.get_mut() = 0;
    }
    fn lookup<'a>(&'a self, key: &Spec::State, _: SearchHandle<Spec>)
            -> Option<&'a SearchNode<Spec>> {
        let my_hash = key.hash();
        let mut posn = my_hash as usize & self.mask;
        for inc in 1..(PROBE_LIMIT + 1) {
            let entry = unsafe { self.arr.get_unchecked(posn) };
            let key_here = entry.k.load(Ordering::Relaxed);
            if key_here == my_hash {
                return convert(entry.v.load(Ordering::Relaxed));
            }
            if key_here == 0 {
                return None;
            }
            posn += inc;
            posn &= self.mask;
        }
        None
    }
}
