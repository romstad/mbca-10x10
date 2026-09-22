//! Move-ordering statistics: butterfly history, capture history, continuation
//! history, counter moves, pawn-structure history, and the three correction
//! histories that adjust the static evaluation.
//!
//! The tables are the conventional set, rescaled to a 10x10 board with eight

use crate::board::Board;
use crate::types::*;

pub const MAX_HISTORY: i32 = 16384;

/// Exponential-decay update.
#[inline]
fn gravity(entry: &mut i16, bonus: i32) {
    let b = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
    let v = *entry as i32;
    *entry = (v + b - v * b.abs() / MAX_HISTORY) as i16;
}

/// Number of pawn-structure buckets.
pub const PAWN_HIST_SIZE: usize = 512;
/// Correction-history buckets, per side.
pub const CORR_SIZE: usize = 16384;
/// Continuation-history planes: one per (piece, destination) pair.
pub const CONT_SIZE: usize = 16 * 100;

/// What a null move (or a stack entry before the root) records instead of a
/// plane.
pub const NULL_CONT: usize = CONT_SIZE;

/// One continuation-history plane: `[moved piece][destination]`.
pub type ContPlane = [[i16; 100]; 16];

/// Fixed-point grain of the correction tables, and their clamp.
const CORR_GRAIN: i32 = 256;
const CORR_MAX: i32 = CORR_GRAIN * 32;

/// The victim axis of capture history.
///
/// `King` is the "no victim" column: a promotion that captures nothing is still
/// a noisy move and still wants its own statistics, and it must not share a cell
/// with a pawn capture on the same square.
#[inline(always)]
pub fn victim_axis(captured: Piece) -> PieceType {
    if captured.is_none() {
        PieceType::King
    } else {
        captured.piece_type()
    }
}

/// The continuation-history plane a move leads to.
#[inline(always)]
pub fn cont_index(p: Piece, to: Square) -> usize {
    debug_assert!(!p.is_none(), "cont_index on Piece::NONE");
    p.idx() * 100 + to.idx()
}

/// The piece half of a `cont_index`.
#[inline(always)]
pub fn cont_piece(i: usize) -> usize {
    i / 100
}

/// The destination half of a `cont_index`.
#[inline(always)]
pub fn cont_to(i: usize) -> usize {
    i % 100
}

#[inline(always)]
pub fn pawn_bucket(pawn_key: u64) -> usize {
    (pawn_key as usize) & (PAWN_HIST_SIZE - 1)
}

pub struct Histories {
    /// `[side to move][from][to]`
    pub main: Box<[[[i16; 100]; 100]; 2]>,
    /// `[moved piece][to][victim]`
    pub capture: Box<[[[i16; 8]; 100]; 16]>,
    /// `[cont_index(prev piece, prev to)][moved piece][to]`
    ///
    /// Exactly `CONT_SIZE` planes: [`NULL_CONT`] has none, so a missed guard
    /// is an index panic rather than plausible statistics for a plane that
    /// means nothing.
    pub cont: Box<[ContPlane; CONT_SIZE]>,
    /// `[prev piece][prev to]`
    pub counter: Box<[[Move; 100]; 16]>,
    /// Quiet history conditioned on the pawn structure:
    /// `[pawn key bucket][moved piece][to]`
    pub pawn: Box<[[[i16; 100]; 16]; PAWN_HIST_SIZE]>,
    /// Pawn-structure correction of the static evaluation, per side.
    pub pawn_corr: Box<[[i32; CORR_SIZE]; 2]>,
    /// Correction keyed on the move that led here, per side.
    pub cont_corr: Box<[[i32; 2]; CONT_SIZE]>,
}

impl Default for Histories {
    fn default() -> Histories {
        Histories::new()
    }
}

impl Histories {
    pub fn new() -> Histories {
        Histories {
            main: vec![[[0i16; 100]; 100]; 2]
                .into_boxed_slice()
                .try_into()
                .ok()
                .unwrap(),
            capture: vec![[[0i16; 8]; 100]; 16]
                .into_boxed_slice()
                .try_into()
                .ok()
                .unwrap(),
            cont: vec![[[0i16; 100]; 16]; CONT_SIZE]
                .into_boxed_slice()
                .try_into()
                .ok()
                .unwrap(),
            counter: vec![[Move::NONE; 100]; 16]
                .into_boxed_slice()
                .try_into()
                .ok()
                .unwrap(),
            pawn: vec![[[0i16; 100]; 16]; PAWN_HIST_SIZE]
                .into_boxed_slice()
                .try_into()
                .ok()
                .unwrap(),
            pawn_corr: vec![[0i32; CORR_SIZE]; 2]
                .into_boxed_slice()
                .try_into()
                .ok()
                .unwrap(),
            cont_corr: vec![[0i32; 2]; CONT_SIZE]
                .into_boxed_slice()
                .try_into()
                .ok()
                .unwrap(),
        }
    }

    /// Zeroes everything, in place. Every loop here is deliberate: an
    /// assignment of an array literal would build it on the stack first.
    pub fn clear(&mut self) {
        for side in self.main.iter_mut() {
            for row in side.iter_mut() {
                row.fill(0);
            }
        }
        for piece in self.capture.iter_mut() {
            for row in piece.iter_mut() {
                row.fill(0);
            }
        }
        for plane in self.cont.iter_mut() {
            for row in plane.iter_mut() {
                row.fill(0);
            }
        }
        for piece in self.counter.iter_mut() {
            piece.fill(Move::NONE);
        }
        for bucket in self.pawn.iter_mut() {
            for row in bucket.iter_mut() {
                row.fill(0);
            }
        }
        for side in self.pawn_corr.iter_mut() {
            side.fill(0);
        }
        for plane in self.cont_corr.iter_mut() {
            plane.fill(0);
        }
    }

    // -- butterfly history ---------------------------------------------------

    #[inline(always)]
    pub fn main_score(&self, c: Color, m: Move) -> i32 {
        self.main[c.idx()][m.from().idx()][m.to().idx()] as i32
    }

    #[inline(always)]
    pub fn update_main(&mut self, c: Color, m: Move, bonus: i32) {
        gravity(&mut self.main[c.idx()][m.from().idx()][m.to().idx()], bonus);
    }

    // -- pawn-structure history ----------------------------------------------

    #[inline(always)]
    pub fn pawn_score(&self, pawn_key: u64, p: Piece, to: Square) -> i32 {
        self.pawn[pawn_bucket(pawn_key)][p.idx()][to.idx()] as i32
    }

    #[inline(always)]
    pub fn update_pawn(&mut self, pawn_key: u64, p: Piece, to: Square, bonus: i32) {
        gravity(
            &mut self.pawn[pawn_bucket(pawn_key)][p.idx()][to.idx()],
            bonus,
        );
    }

    // -- capture history -----------------------------------------------------

    #[inline(always)]
    pub fn capture_score(&self, p: Piece, to: Square, victim: PieceType) -> i32 {
        self.capture[p.idx()][to.idx()][victim.idx()] as i32
    }

    #[inline(always)]
    pub fn update_capture(&mut self, p: Piece, to: Square, victim: PieceType, bonus: i32) {
        gravity(&mut self.capture[p.idx()][to.idx()][victim.idx()], bonus);
    }

    // -- continuation history ------------------------------------------------

    #[inline(always)]
    pub fn cont_score(&self, plane: usize, p: Piece, to: Square) -> i32 {
        self.cont[plane][p.idx()][to.idx()] as i32
    }

    #[inline(always)]
    pub fn update_cont(&mut self, plane: usize, p: Piece, to: Square, bonus: i32) {
        gravity(&mut self.cont[plane][p.idx()][to.idx()], bonus);
    }

    // -- counter moves -------------------------------------------------------

    #[inline(always)]
    pub fn counter_move(&self, plane: usize) -> Move {
        self.counter[cont_piece(plane)][cont_to(plane)]
    }

    #[inline(always)]
    pub fn set_counter_move(&mut self, plane: usize, m: Move) {
        self.counter[cont_piece(plane)][cont_to(plane)] = m;
    }

    // -- correction history --------------------------------------------------

    /// How much the search has historically disagreed with the static
    /// evaluation in positions that share this pawn structure or this
    /// predecessor move.
    #[inline]
    pub fn correction(&self, b: &Board, prev: usize) -> i32 {
        let c = b.stm.idx();
        let p = self.pawn_corr[c][(b.pawn_key as usize) & (CORR_SIZE - 1)];
        let k = if prev == NULL_CONT {
            0
        } else {
            self.cont_corr[prev][c]
        };
        let (wp, wk) = (crate::params::CORR_PAWN_W, crate::params::CORR_CONT_W);
        let total = (wp + wk).max(1);
        (p * wp + k * wk) / (CORR_GRAIN * total)
    }

    pub fn update_correction(&mut self, b: &Board, prev: usize, diff: i32, depth: i32) {
        let c = b.stm.idx();
        let weight = (depth + 1).min(16);
        let target = diff * CORR_GRAIN;
        let blend = |e: &mut i32| {
            *e = (*e * (256 - weight) + target * weight) / 256;
            *e = (*e).clamp(-CORR_MAX, CORR_MAX);
        };
        blend(&mut self.pawn_corr[c][(b.pawn_key as usize) & (CORR_SIZE - 1)]);
        if prev != NULL_CONT {
            blend(&mut self.cont_corr[prev][c]);
        }
    }
}

/// Continuation-history planes of the preceding moves. Every entry is a valid
/// plane: [`NULL_CONT`] where the corresponding ply held a null move or lies
/// before the root.
pub type Conts = [usize; 4];
