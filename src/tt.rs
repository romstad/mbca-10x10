//! Transposition table.

use crate::search::VALUE_NONE;
use crate::types::Move;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_UPPER: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

const ENTRIES_PER_BUCKET: usize = 4;

/// Width of the move field. `Move` uses 18 bits: `from | to << 7 | promo << 14
/// | ep << 17`.
const MOVE_BITS: u32 = 18;
const MOVE_MASK: u64 = (1 << MOVE_BITS) - 1;

const GEN_SHIFT: u32 = 61;
const GEN_BITS: u32 = 3;
const GEN_MASK: u64 = (1 << GEN_BITS) - 1;
/// One full turn of the generation counter, for the wrap-around in the age term.
const GEN_CYCLE: i32 = 1 << GEN_BITS;

/// `n` empty buckets, straight from the kernel.
fn zeroed_buckets(n: usize) -> Vec<Bucket> {
    let layout = std::alloc::Layout::array::<Bucket>(n).expect("tt: bucket array overflows");
    // Safety: the layout is non-zero (`n >= 1`), the all-zeros bit pattern is a
    // valid `Bucket`, and the layout is exactly the one `Vec` will free with.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout).cast::<Bucket>();
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Vec::from_raw_parts(p, n, n)
    }
}

#[derive(Default)]
#[repr(C, align(64))]
struct Bucket {
    key: [AtomicU64; ENTRIES_PER_BUCKET],
    data: [AtomicU64; ENTRIES_PER_BUCKET],
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: u8,
    pub was_pv: bool,
}

#[inline(always)]
fn pack(mv: Move, score: i16, eval: i16, depth: u8, bound: u8, gen: u8, pv: bool) -> u64 {
    debug_assert!(
        (mv.0 as u64) <= MOVE_MASK,
        "move does not fit the {MOVE_BITS}-bit TT field: {:#x}",
        mv.0
    );
    (mv.0 as u64 & MOVE_MASK)
        | (score as u16 as u64) << 18
        | (eval as u16 as u64) << 34
        | (depth as u64) << 50
        | (bound as u64) << 58
        | (pv as u64) << 60
        | (gen as u64 & GEN_MASK) << GEN_SHIFT
}

#[inline(always)]
fn unpack_move(d: u64) -> Move {
    Move((d & MOVE_MASK) as u32)
}
#[inline(always)]
fn unpack_score(d: u64) -> i32 {
    (((d >> 18) & 0xFFFF) as u16 as i16) as i32
}
#[inline(always)]
fn unpack_eval(d: u64) -> i32 {
    (((d >> 34) & 0xFFFF) as u16 as i16) as i32
}
#[inline(always)]
fn unpack_depth(d: u64) -> i32 {
    ((d >> 50) & 0xFF) as i32
}
#[inline(always)]
fn unpack_bound(d: u64) -> u8 {
    ((d >> 58) & 3) as u8
}
#[inline(always)]
fn unpack_pv(d: u64) -> bool {
    (d >> 60) & 1 != 0
}
#[inline(always)]
fn unpack_gen(d: u64) -> u8 {
    ((d >> GEN_SHIFT) & GEN_MASK) as u8
}

pub struct Tt {
    buckets: Vec<Bucket>,
    mask: usize,
    generation: AtomicU8,
}

impl Tt {
    pub fn new(mb: usize) -> Tt {
        let mut tt = Tt {
            buckets: Vec::new(),
            mask: 0,
            generation: AtomicU8::new(0),
        };
        tt.resize(mb);
        tt
    }

    pub fn resize(&mut self, mb: usize) {
        let bytes = mb.max(1) * 1024 * 1024;
        let mut n = bytes / std::mem::size_of::<Bucket>();
        n = n.max(1).next_power_of_two();
        if n * std::mem::size_of::<Bucket>() > bytes {
            n /= 2;
        }
        let n = n.max(1);
        self.buckets = Vec::new();
        self.buckets = zeroed_buckets(n);
        self.mask = n - 1;
    }

    pub fn clear(&self) {
        for b in &self.buckets {
            for i in 0..ENTRIES_PER_BUCKET {
                b.key[i].store(0, Ordering::Relaxed);
                b.data[i].store(0, Ordering::Relaxed);
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    fn gen(&self) -> u8 {
        (self.generation.load(Ordering::Relaxed) as u64 & GEN_MASK) as u8
    }

    #[inline(always)]
    fn bucket(&self, key: u64) -> &Bucket {
        unsafe { self.buckets.get_unchecked((key as usize) & self.mask) }
    }

    pub fn prefetch(&self, key: u64) {
        // The aarch64 prefetch intrinsic is still unstable, so emit it directly.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let p = self.bucket(key) as *const Bucket;
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                self.bucket(key) as *const Bucket as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = key;
    }

    pub fn probe(&self, key: u64) -> Option<Hit> {
        let b = self.bucket(key);
        for i in 0..ENTRIES_PER_BUCKET {
            let data = b.data[i].load(Ordering::Relaxed);
            if b.key[i].load(Ordering::Relaxed) ^ data == key && data != 0 {
                // Refresh the age so a useful entry survives replacement.
                let gen = self.gen();
                if unpack_gen(data) != gen {
                    let nd = (data & !(GEN_MASK << GEN_SHIFT)) | (gen as u64) << GEN_SHIFT;
                    b.data[i].store(nd, Ordering::Relaxed);
                    b.key[i].store(key ^ nd, Ordering::Relaxed);
                }
                return Some(Hit {
                    mv: unpack_move(data),
                    score: unpack_score(data),
                    eval: unpack_eval(data),
                    depth: unpack_depth(data),
                    bound: unpack_bound(data),
                    was_pv: unpack_pv(data),
                });
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        key: u64,
        mv: Move,
        score: i32,
        eval: i32,
        depth: i32,
        bound: u8,
        was_pv: bool,
    ) {
        // `Move::NULL` is `u32::MAX`; stored, it would truncate to a valid-
        // looking 18-bit move belonging to no position at all.
        debug_assert!(
            mv != Move::NULL,
            "Move::NULL reached the transposition table"
        );
        let b = self.bucket(key);
        let gen = self.gen();
        let mut slot = 0usize;
        let mut worst = i32::MAX;
        for i in 0..ENTRIES_PER_BUCKET {
            let data = b.data[i].load(Ordering::Relaxed);
            if data == 0 || b.key[i].load(Ordering::Relaxed) ^ data == key {
                slot = i;
                // Keep a deeper entry for the same position unless this one is
                // exact or noticeably deeper.
                if data != 0
                    && bound != BOUND_EXACT
                    && depth + 4 < unpack_depth(data)
                    && unpack_gen(data) == gen
                {
                    return;
                }
                break;
            }
            // Prefer to overwrite shallow and stale entries.
            let age = ((gen as i32 + GEN_CYCLE - unpack_gen(data) as i32) & GEN_MASK as i32) * 3;
            let value = unpack_depth(data) - age;
            if value < worst {
                worst = value;
                slot = i;
            }
        }
        let mv = if mv.is_none() {
            let data = b.data[slot].load(Ordering::Relaxed);
            if b.key[slot].load(Ordering::Relaxed) ^ data == key {
                unpack_move(data)
            } else {
                mv
            }
        } else {
            mv
        };
        let data = pack(
            mv,
            // `VALUE_NONE` is an eval-only sentinel, not a mate score. It
            // must survive a store so readers cannot mistake it for `MATE`.
            score.clamp(-VALUE_NONE, VALUE_NONE) as i16,
            eval.clamp(-VALUE_NONE, VALUE_NONE) as i16,
            depth.clamp(0, 255) as u8,
            bound,
            gen,
            was_pv,
        );
        b.data[slot].store(data, Ordering::Relaxed);
        b.key[slot].store(key ^ data, Ordering::Relaxed);
    }

    /// Permille of the table that has been written this generation.
    pub fn hashfull(&self) -> usize {
        let gen = self.gen();
        let mut used = 0;
        let n = 1000.min(self.buckets.len());
        for b in self.buckets.iter().take(n) {
            for i in 0..ENTRIES_PER_BUCKET {
                let data = b.data[i].load(Ordering::Relaxed);
                if data != 0 && unpack_gen(data) == gen {
                    used += 1;
                }
            }
        }
        used * 1000 / (n * ENTRIES_PER_BUCKET)
    }
}
