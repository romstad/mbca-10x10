//! Fully legal move generation.
//!
//! Moves are generated straight into a legal list: targets are restricted by the
//! check mask, pinned pieces are held on their pin ray, and king moves and en
//! passant get an explicit safety test.
//!
//! The pawn section is where Grand Chess actually differs. An orthodox
//! generator splits pawns into "promoting" and "plain" by their *source* rank
//! and pushes the staging filter into the destination masks. Neither survives here:
//! promotion is optional on two ranks and mandatory on a third, the outcome depends
//! on the side's promotion inventory, and a move from rank 8 to rank 9 is a promotion
//! opportunity even though it starts inside the zone. So destinations are
//! computed against the check mask alone and classified one at a time, by
//! destination rank, in `add_pawn`.

use crate::attacks::*;
use crate::board::Board;
use crate::squareset::SquareSet;
use crate::types::*;
use std::mem::MaybeUninit;

pub const GEN_ALL: u8 = 0;
pub const GEN_CAPTURES: u8 = 1;
pub const GEN_QUIETS: u8 = 2;

/// Does this move win material outright: capture, en passant, or promotion?
#[inline(always)]
pub fn is_capture(b: &Board, m: Move) -> bool {
    !b.piece_at(m.to()).is_none() || m.is_en_passant()
}

/// The complement of [`is_capture`] within non-promoting moves, i.e. exactly
/// what `GEN_QUIETS` generates.
#[inline(always)]
pub fn is_quiet(b: &Board, m: Move) -> bool {
    !m.is_promotion() && !is_capture(b, m)
}

/// Maximum number of legal moves. Should be safe?
pub const MAX_MOVES: usize = 256;

/// A fixed-capacity move list.
///
/// The arrays are deliberately left uninitialised: a move list is built once per
/// node, and zeroing 2 kB every time is real time on the clock. Only entries
/// below `len` are ever read, and every one of those has been written.
pub struct MoveList {
    moves: [MaybeUninit<Move>; MAX_MOVES],
    scores: [MaybeUninit<i32>; MAX_MOVES],
    pub len: usize,
}

impl Default for MoveList {
    fn default() -> MoveList {
        MoveList::new()
    }
}

impl MoveList {
    pub fn new() -> MoveList {
        MoveList {
            // An array of `MaybeUninit` is itself always initialised.
            moves: unsafe { MaybeUninit::uninit().assume_init() },
            scores: unsafe { MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }
    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        unsafe {
            self.moves.get_unchecked_mut(self.len).write(m);
        }
        self.len += 1;
    }
    #[inline(always)]
    pub fn get(&self, i: usize) -> Move {
        debug_assert!(i < self.len);
        unsafe { self.moves.get_unchecked(i).assume_init() }
    }
    #[inline(always)]
    pub fn score(&self, i: usize) -> i32 {
        debug_assert!(i < self.len);
        unsafe { self.scores.get_unchecked(i).assume_init() }
    }
    #[inline(always)]
    pub fn set_score(&mut self, i: usize, s: i32) {
        debug_assert!(i < self.len);
        unsafe {
            self.scores.get_unchecked_mut(i).write(s);
        }
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        // Every entry below `len` has been written by `push`.
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr() as *const Move, self.len) }
    }
    /// Selection sort step: bring the best remaining move to `start`.
    #[inline]
    pub fn pick_best(&mut self, start: usize) -> Move {
        let mut best = start;
        let mut best_score = self.score(start);
        for i in start + 1..self.len {
            let s = self.score(i);
            if s > best_score {
                best = i;
                best_score = s;
            }
        }
        self.moves.swap(start, best);
        self.scores.swap(start, best);
        self.get(start)
    }
}

#[inline(always)]
fn shift_capture(ss: SquareSet, c: Color, east: bool) -> SquareSet {
    match (c, east) {
        (Color::White, true) => ss.north_east(),
        (Color::White, false) => ss.north_west(),
        (Color::Black, true) => ss.south_east(),
        (Color::Black, false) => ss.south_west(),
    }
}

/// Index delta added to `from` to reach `to` for a single pawn push.
#[inline(always)]
fn forward_delta(c: Color) -> i32 {
    match c {
        Color::White => 10,
        Color::Black => -10,
    }
}

#[inline(always)]
fn capture_delta(c: Color, east: bool) -> i32 {
    match (c, east) {
        (Color::White, true) => 11,
        (Color::White, false) => 9,
        (Color::Black, true) => -9,
        (Color::Black, false) => -11,
    }
}

/// Pawns that can reach the promotion zone in one move: relative ranks 6, 7 and
/// 8, which move to relative ranks 7, 8 and 9.
#[inline(always)]
fn zone_sources(c: Color) -> SquareSet {
    SquareSet::relative_rank(c, 6) | SquareSet::relative_rank(c, 7) | SquareSet::relative_rank(c, 8)
}

/// Emits every legal outcome of one pawn move, given the side's promotion
/// inventory as a bitmask over `PieceType::idx()`.
#[inline(always)]
fn add_pawn<const T: u8>(
    list: &mut MoveList,
    from: Square,
    to: Square,
    us: Color,
    inv: u8,
    capture: bool,
) {
    let rr = to.relative_rank(us);
    if rr <= 8 {
        let wanted = if capture {
            T != GEN_QUIETS
        } else {
            T != GEN_CAPTURES
        };
        if wanted {
            list.push(Move::new(from, to));
        }
    }
    if rr >= 7 && T != GEN_QUIETS {
        for pt in PROMO_TYPES {
            if inv & (1u8 << pt.idx()) != 0 {
                list.push(Move::promotion(from, to, pt));
            }
        }
    }
}

pub fn generate<const T: u8>(b: &Board, list: &mut MoveList) {
    let us = b.stm;
    let them = us.flip();
    let occ = b.occupied();
    let ours = b.colors(us);
    let theirs = b.colors(them);
    let ksq = b.king_sq(us);
    let blockers = b.blockers[us.idx()];

    // --- king moves --------------------------------------------------------
    let mut king_targets = king_attacks(ksq) - ours;
    if T == GEN_CAPTURES {
        king_targets &= theirs;
    } else if T == GEN_QUIETS {
        king_targets -= theirs;
    }
    // The king is removed from the occupancy so it cannot shield itself along a
    // ray it is fleeing down.
    let occ_no_king = occ - SquareSet::from_square(ksq);
    for to in king_targets {
        if !b.attacked_by(them, to, occ_no_king) {
            list.push(Move::new(ksq, to));
        }
    }

    if b.checkers.more_than_one() {
        return; // Only the king can move out of a double check.
    }

    // --- target squares for every other piece ------------------------------
    let base_target = if b.checkers.any() {
        between(ksq, b.checkers.lsb()) | b.checkers
    } else {
        !ours
    };
    let mut target = base_target;
    if T == GEN_CAPTURES {
        target &= theirs;
    } else if T == GEN_QUIETS {
        target -= theirs;
    }

    // --- pawns -------------------------------------------------------------
    let pawns = b.colored(us, PieceType::Pawn);
    let empty = !occ;
    let zone_pawns = pawns & zone_sources(us);
    let inv = if zone_pawns.any() {
        b.promo_inventory(us)
    } else {
        0
    };

    // Under GEN_CAPTURES a push is worth generating only if it can promote.
    if T != GEN_CAPTURES || inv != 0 {
        let d = forward_delta(us);
        let one = pawns.forward(us) & empty;
        for to in one & base_target {
            let from = Square((to.0 as i32 - d) as u8);
            if !blockers.has(from) || aligned(ksq, from, to) {
                add_pawn::<T>(list, from, to, us, inv, false);
            }
        }
        if T != GEN_CAPTURES {
            let two = (one & SquareSet::relative_rank(us, 3)).forward(us) & empty;
            for to in two & base_target {
                let from = Square((to.0 as i32 - 2 * d) as u8);
                if !blockers.has(from) || aligned(ksq, from, to) {
                    list.push(Move::new(from, to));
                }
            }
        }
    }

    for east in [true, false] {
        let d = capture_delta(us, east);
        for to in shift_capture(pawns, us, east) & theirs & base_target {
            let from = Square((to.0 as i32 - d) as u8);
            if !blockers.has(from) || aligned(ksq, from, to) {
                add_pawn::<T>(list, from, to, us, inv, true);
            }
        }
    }

    // En passant needs its own legality test.
    if T != GEN_QUIETS && b.ep != Square::NONE {
        let ep = b.ep;
        debug_assert_eq!(ep.relative_rank(us), 6);
        for from in pawn_attacks(them, ep) & pawns {
            let m = Move::en_passant(from, ep);
            if b.is_legal(m) {
                list.push(m);
            }
        }
    }

    // --- knights, bishops, rooks, queens, marshals, cardinals ---------------
    for pt in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::Marshal,
        PieceType::Cardinal,
    ] {
        for from in b.colored(us, pt) {
            let mut to_set = piece_attacks(pt, from, occ) & target;
            if blockers.has(from) {
                to_set &= line(ksq, from);
            }
            for to in to_set {
                list.push(Move::new(from, to));
            }
        }
    }
}
