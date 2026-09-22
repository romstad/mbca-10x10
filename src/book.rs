//! The opening book: the file format, the runtime probe, and the tools.
//!
//! The book exists to make games **varied**, not strong. Grand Chess has no
//! opening theory to draw on, and an engine with a deterministic search plays
//! the same game every time; a book is where the variety comes from.

use crate::board::Board;
use crate::movegen::{generate, MoveList, GEN_ALL};
use crate::rng::Rng;
use crate::search::MATE;
use crate::types::{move_to_string, Move};

pub const ENTRY_SIZE: usize = 16;
pub const HEADER_SIZE: usize = 32;
pub const BOOK_MAGIC: [u8; 8] = *b"MBCABOOK";
pub const BOOK_VERSION: u16 = 1;

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Entry {
    pub key: u64,
    pub mv: u32,
    pub weight: u16,
    pub score: i16,
}

const _: () = assert!(std::mem::size_of::<Entry>() == ENTRY_SIZE);

impl Entry {
    pub fn from_bytes(b: &[u8]) -> Entry {
        Entry {
            key: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            mv: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            weight: u16::from_le_bytes(b[12..14].try_into().unwrap()),
            score: i16::from_le_bytes(b[14..16].try_into().unwrap()),
        }
    }

    pub fn mov(&self) -> Move {
        Move(self.mv)
    }

    /// Whether `mv` could be a move at all, ignoring the position.
    pub fn valid_move(&self) -> bool {
        let m = self.mov();
        self.mv >> 18 == 0
            && self.mv != 0
            && m.from().0 < 100
            && m.to().0 < 100
            && m.promo_bits() <= 6
    }

    /// The sort key: ascending by position, then by move.
    fn order(&self) -> (u64, u32) {
        (self.key, self.mv)
    }
}

// ---------------------------------------------------------------------------
// The book
// ---------------------------------------------------------------------------

pub struct Book {
    entries: Vec<Entry>,
}

impl Book {
    /// Validates and sorts a set of entries into a book.
    pub fn build(mut entries: Vec<Entry>) -> Result<Book, String> {
        entries.sort_unstable_by_key(|e| e.order());
        for w in entries.windows(2) {
            if w[0].order() == w[1].order() {
                return Err(format!(
                    "duplicate entry for key {:016x} move {}",
                    w[0].key,
                    move_to_string(w[0].mov())
                ));
            }
        }
        let book = Book { entries };
        book.check_entries()?;
        Ok(book)
    }

    fn check_entries(&self) -> Result<(), String> {
        for e in &self.entries {
            if !e.valid_move() {
                return Err(format!(
                    "key {:016x} has {:#x}, which is not a move",
                    e.key, e.mv
                ));
            }
            if (e.score as i32) < -MATE || e.score as i32 > MATE {
                return Err(format!("key {:016x} has score {}", e.key, e.score));
            }
        }
        Ok(())
    }

    /// Decodes a book, rejecting anything this binary must not trust.
    pub fn from_bytes(bytes: &[u8]) -> Result<Book, String> {
        if bytes.len() < HEADER_SIZE {
            return Err(format!("{} bytes is shorter than a header", bytes.len()));
        }
        if bytes[0..8] != BOOK_MAGIC {
            return Err("not a book file (bad magic)".to_string());
        }
        let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        if version != BOOK_VERSION {
            return Err(format!("version {version}, expected {BOOK_VERSION}"));
        }
        let entry_size = u16::from_le_bytes(bytes[10..12].try_into().unwrap());
        if entry_size as usize != ENTRY_SIZE {
            return Err(format!("entry size {entry_size}, expected {ENTRY_SIZE}"));
        }
        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if flags != 0 {
            return Err(format!("reserved flags are {flags:#x}, expected 0"));
        }
        let count = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        let expect = count
            .checked_mul(ENTRY_SIZE)
            .and_then(|n| n.checked_add(HEADER_SIZE));
        if expect != Some(bytes.len()) {
            return Err(format!(
                "header claims {count} entries ({}) but the file is {} bytes",
                expect.map_or_else(
                    || "more bytes than fit in a `usize`".to_string(),
                    |n| format!("{n} bytes")
                ),
                bytes.len()
            ));
        }
        let entries: Vec<Entry> = bytes[HEADER_SIZE..]
            .chunks_exact(ENTRY_SIZE)
            .map(Entry::from_bytes)
            .collect();
        for (i, w) in entries.windows(2).enumerate() {
            if w[0].order() >= w[1].order() {
                return Err(format!(
                    "entries {i} and {} are out of order or duplicated",
                    i + 1
                ));
            }
        }
        let book = Book { entries };
        book.check_entries()?;
        Ok(book)
    }

    pub fn load(path: &str) -> Result<Book, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Book::from_bytes(&bytes).map_err(|e| format!("{path}: {e}"))
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// How many distinct positions the book covers.
    pub fn positions(&self) -> usize {
        let mut n = 0;
        let mut last = None;
        for e in &self.entries {
            if last != Some(e.key) {
                n += 1;
                last = Some(e.key);
            }
        }
        n
    }

    /// Every entry for `key`, in file order. Empty on a miss.
    pub fn get(&self, key: u64) -> &[Entry] {
        let lo = self.entries.partition_point(|e| e.key < key);
        let hi = lo
            + self.entries[lo..]
                .iter()
                .take_while(|e| e.key == key)
                .count();
        &self.entries[lo..hi]
    }

    /// The entries for `b`, split by whether their move is actually legal in it.
    pub fn probe(&self, b: &Board) -> Probe {
        let run = self.get(b.key);
        let mut list = MoveList::new();
        generate::<GEN_ALL>(b, &mut list);
        let legal_here = |m: Move| (0..list.len).any(|i| list.get(i) == m);
        let mut legal = Vec::with_capacity(run.len());
        let mut illegal = Vec::new();
        for &e in run {
            if e.valid_move() && legal_here(e.mov()) {
                legal.push(e);
            } else {
                illegal.push(e);
            }
        }
        Probe { legal, illegal }
    }
}

/// The result of looking one position up.
pub struct Probe {
    /// Entries whose move is legal in the position, in file order.
    pub legal: Vec<Entry>,
    /// Entries whose move is not. Never non-empty without a bug somewhere.
    pub illegal: Vec<Entry>,
}

/// The move a probe chose, and enough context to explain the choice.
pub struct Choice {
    pub entry: Entry,
    /// The entry's weight after `variety` was applied.
    pub weight: u32,
    /// The sum over the run, so `weight / total` is the probability.
    pub total: u32,
    /// How many legal entries were in play.
    pub moves: usize,
}

/// Applies `BookVariety` to a run of weights.
///
/// `variety` is a percentage of the emitted temperature. 100 leaves the weights
/// exactly as the book has them. 0 is deterministic: all weight on the best
/// move. Above 100 flattens, `w' = (w / w_max) ^ (100 / variety) * w_max`, which
/// pulls every weight toward the largest as `variety` grows; so 400 plays the
/// fourth-choice move about as often as the first, while still never playing a
/// weight-0 move.
fn apply_variety(weights: &[u16], variety: i32) -> Vec<u32> {
    let w_max = weights.iter().copied().max().unwrap_or(0);
    if w_max == 0 {
        return vec![0; weights.len()];
    }
    if variety <= 0 {
        let best = weights.iter().position(|&w| w == w_max).unwrap();
        let mut out = vec![0u32; weights.len()];
        out[best] = 1;
        return out;
    }
    if variety == 100 {
        return weights.iter().map(|&w| w as u32).collect();
    }
    let exp = 100.0 / variety as f64;
    weights
        .iter()
        .map(|&w| {
            if w == 0 {
                return 0;
            }
            let r = (w as f64 / w_max as f64).powf(exp);
            ((r * w_max as f64).round() as u32).max(1)
        })
        .collect()
}

/// Chooses one entry by weight. `None` when nothing is playable.
pub fn choose(legal: &[Entry], variety: i32, rng: &mut Rng) -> Option<Choice> {
    let weights: Vec<u16> = legal.iter().map(|e| e.weight).collect();
    let eff = apply_variety(&weights, variety);
    let total: u32 = eff.iter().sum();
    if total == 0 {
        return None;
    }
    let mut r = rng.below(total as u64) as u32;
    for (i, &w) in eff.iter().enumerate() {
        if r < w {
            return Some(Choice {
                entry: legal[i],
                weight: w,
                total,
                moves: legal.len(),
            });
        }
        r -= w;
    }
    unreachable!("the cumulative weights must cover the draw")
}

/// `cp N` or `mate N`, for the score of one book entry.
pub fn score_string(score: i16) -> String {
    crate::search::score_to_uci(score as i32)
}
