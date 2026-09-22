//! Staged move ordering.
//!
//! Moves come out in the order a good search wants to see them: the hash move,
//! the winning captures and promotions, the two killers, the counter move, the
//! quiets by history, and finally the losing captures. Generation is deferred
//! stage by stage, so a node that fails high on its hash move never builds a
//! move list at all.

use std::mem::MaybeUninit;

use crate::board::Board;
use crate::history::{victim_axis, Conts, Histories};
use crate::movegen::*;
use crate::params::VAL_PAWN;
use crate::see::{see_ge, see_value};
use crate::types::*;

const STAGE_TT: u8 = 0;
const STAGE_GEN_CAPTURES: u8 = 1;
const STAGE_GOOD_CAPTURES: u8 = 2;
const STAGE_KILLER_1: u8 = 3;
const STAGE_KILLER_2: u8 = 4;
const STAGE_COUNTER: u8 = 5;
const STAGE_GEN_QUIETS: u8 = 6;
const STAGE_QUIETS: u8 = 7;
const STAGE_BAD_CAPTURES: u8 = 8;
const STAGE_DONE: u8 = 9;
const STAGE_EVASION_GEN: u8 = 10;
const STAGE_EVASIONS: u8 = 11;

/// Losing captures, deferred to the end.
const MAX_BAD: usize = 128;

/// Static value of what a move wins, before history: the victim, plus the
/// upgrade if it promotes.
#[inline]
pub fn mvv(b: &Board, m: Move) -> i32 {
    let victim = if m.is_en_passant() {
        VAL_PAWN
    } else {
        let p = b.piece_at(m.to());
        if p.is_none() {
            0
        } else {
            see_value(p.piece_type())
        }
    };
    let promo = if m.is_promotion() {
        see_value(m.promo()) - VAL_PAWN
    } else {
        0
    };
    victim + promo
}

/// The capture-history cell a move belongs in: `[mover][to][victim]`, with
/// `King` standing for "captured nothing".
#[inline]
pub fn capture_cell(b: &Board, m: Move) -> (Piece, Square, PieceType) {
    let victim = if m.is_en_passant() {
        PieceType::Pawn
    } else {
        victim_axis(b.piece_at(m.to()))
    };
    (b.piece_at(m.from()), m.to(), victim)
}

pub struct MovePicker {
    stage: u8,
    tt_move: Move,
    killers: [Move; 2],
    counter: Move,
    list: MoveList,
    bad: [MaybeUninit<Move>; MAX_BAD],
    n_bad: usize,
    idx: usize,
    bad_idx: usize,
    /// Quiescence and ProbCut: captures and promotions only, no quiet stages.
    noisy_only: bool,
    see_threshold: i32,
}

impl MovePicker {
    pub fn new(tt_move: Move, killers: [Move; 2], counter: Move, in_check: bool) -> MovePicker {
        MovePicker {
            stage: if in_check {
                STAGE_EVASION_GEN
            } else {
                STAGE_TT
            },
            tt_move,
            killers,
            counter,
            list: MoveList::new(),
            // An array of `MaybeUninit` is itself always initialised.
            bad: unsafe { MaybeUninit::uninit().assume_init() },
            n_bad: 0,
            idx: 0,
            bad_idx: 0,
            noisy_only: false,
            see_threshold: 0,
        }
    }

    pub fn new_qsearch(tt_move: Move, in_check: bool) -> MovePicker {
        let mut mp = MovePicker::new(tt_move, [Move::NONE; 2], Move::NONE, in_check);
        mp.noisy_only = true;
        mp
    }

    /// Captures and promotions winning at least `threshold`, for ProbCut.
    pub fn new_probcut(tt_move: Move, threshold: i32) -> MovePicker {
        let mut mp = MovePicker::new(tt_move, [Move::NONE; 2], Move::NONE, false);
        mp.noisy_only = true;
        mp.see_threshold = threshold;
        mp
    }

    fn score_captures(&mut self, b: &Board, h: &Histories) {
        for i in 0..self.list.len {
            let m = self.list.get(i);
            let (p, to, victim) = capture_cell(b, m);
            let s = mvv(b, m) * 16 + h.capture_score(p, to, victim);
            self.list.set_score(i, s);
        }
    }

    fn quiet_score(b: &Board, h: &Histories, conts: &Conts, m: Move) -> i32 {
        let p = b.piece_at(m.from());
        let mut s = h.main_score(b.stm, m) + h.pawn_score(b.pawn_key, p, m.to());
        for &c in conts.iter() {
            if c != crate::history::NULL_CONT {
                s += h.cont_score(c, p, m.to());
            }
        }
        s
    }

    fn score_quiets(&mut self, b: &Board, h: &Histories, conts: &Conts) {
        for i in 0..self.list.len {
            let s = MovePicker::quiet_score(b, h, conts, self.list.get(i));
            self.list.set_score(i, s);
        }
    }

    /// In check the whole list is generated at once, so noisy and quiet
    /// evasions are scored on one scale: anything that wins material first, the
    /// rest by quiet history.
    fn score_evasions(&mut self, b: &Board, h: &Histories, conts: &Conts) {
        for i in 0..self.list.len {
            let m = self.list.get(i);
            let s = if is_quiet(b, m) {
                MovePicker::quiet_score(b, h, conts, m)
            } else {
                let (p, to, victim) = capture_cell(b, m);
                (1 << 22) + mvv(b, m) * 16 + h.capture_score(p, to, victim)
            };
            self.list.set_score(i, s);
        }
    }

    /// A hash move is worth trying only if it is legal here *and*, in a
    /// noisy-only search, noisy enough. `is_pseudo_legal` comes first because
    /// everything after it reads the board at the move's squares.
    fn tt_move_usable(&self, b: &Board) -> bool {
        let m = self.tt_move;
        if m.is_none() || !b.is_pseudo_legal(m) {
            return false;
        }
        !self.noisy_only || (!is_quiet(b, m) && see_ge(b, m, self.see_threshold))
    }

    /// A killer or counter move is only ever tried in the quiet stages, so it
    /// has to still be quiet in *this* position; not merely quiet when it was
    /// stored.
    fn quiet_candidate(&self, b: &Board, m: Move) -> bool {
        !m.is_none() && m != self.tt_move && b.is_pseudo_legal(m) && is_quiet(b, m)
    }

    /// Whether the move just yielded came from the losing-capture stage.
    ///
    /// True only while that stage is handing moves out, so every move it
    /// describes is a capture or promotion the SEE test rejected. It is false
    /// for evasions, which never reach the stage, and in the quiescence and
    /// ProbCut pickers, which leave `STAGE_GOOD_CAPTURES` for `STAGE_DONE`.
    #[inline(always)]
    pub fn in_bad_captures(&self) -> bool {
        self.stage == STAGE_BAD_CAPTURES
    }

    pub fn next(
        &mut self,
        b: &Board,
        h: &Histories,
        conts: &Conts,
        skip_quiets: bool,
    ) -> Option<Move> {
        loop {
            match self.stage {
                STAGE_TT => {
                    self.stage = STAGE_GEN_CAPTURES;
                    if self.tt_move_usable(b) {
                        return Some(self.tt_move);
                    }
                }
                STAGE_GEN_CAPTURES => {
                    self.list.clear();
                    generate::<GEN_CAPTURES>(b, &mut self.list);
                    self.score_captures(b, h);
                    self.idx = 0;
                    self.stage = STAGE_GOOD_CAPTURES;
                }
                STAGE_GOOD_CAPTURES => {
                    while self.idx < self.list.len {
                        let m = self.list.pick_best(self.idx);
                        self.idx += 1;
                        if m == self.tt_move {
                            continue;
                        }
                        if see_ge(b, m, self.see_threshold) {
                            return Some(m);
                        }
                        if self.n_bad < MAX_BAD {
                            unsafe { self.bad.get_unchecked_mut(self.n_bad).write(m) };
                            self.n_bad += 1;
                        }
                    }
                    self.stage = if self.noisy_only {
                        STAGE_DONE
                    } else {
                        STAGE_KILLER_1
                    };
                }
                STAGE_KILLER_1 | STAGE_KILLER_2 => {
                    let k = self.killers[(self.stage - STAGE_KILLER_1) as usize];
                    self.stage += 1;
                    if skip_quiets {
                        self.stage = STAGE_BAD_CAPTURES;
                        self.bad_idx = 0;
                        continue;
                    }
                    if self.stage == STAGE_COUNTER && k == self.killers[0] {
                        continue;
                    }
                    if self.quiet_candidate(b, k) {
                        return Some(k);
                    }
                }
                STAGE_COUNTER => {
                    let c = self.counter;
                    self.stage = STAGE_GEN_QUIETS;
                    if skip_quiets {
                        self.stage = STAGE_BAD_CAPTURES;
                        self.bad_idx = 0;
                        continue;
                    }
                    if c != self.killers[0] && c != self.killers[1] && self.quiet_candidate(b, c) {
                        return Some(c);
                    }
                }
                STAGE_GEN_QUIETS => {
                    self.list.clear();
                    if !skip_quiets {
                        generate::<GEN_QUIETS>(b, &mut self.list);
                        self.score_quiets(b, h, conts);
                    }
                    self.idx = 0;
                    self.stage = STAGE_QUIETS;
                }
                STAGE_QUIETS => {
                    if !skip_quiets {
                        while self.idx < self.list.len {
                            let m = self.list.pick_best(self.idx);
                            self.idx += 1;
                            if m == self.tt_move
                                || m == self.killers[0]
                                || m == self.killers[1]
                                || m == self.counter
                            {
                                continue;
                            }
                            return Some(m);
                        }
                    }
                    self.stage = STAGE_BAD_CAPTURES;
                    self.bad_idx = 0;
                }
                STAGE_BAD_CAPTURES => {
                    if self.bad_idx < self.n_bad {
                        let m = unsafe { self.bad.get_unchecked(self.bad_idx).assume_init() };
                        self.bad_idx += 1;
                        return Some(m);
                    }
                    self.stage = STAGE_DONE;
                }
                STAGE_EVASION_GEN => {
                    self.list.clear();
                    generate::<GEN_ALL>(b, &mut self.list);
                    self.score_evasions(b, h, conts);
                    for i in 0..self.list.len {
                        if self.list.get(i) == self.tt_move {
                            self.list.set_score(i, 1 << 28);
                        }
                    }
                    self.idx = 0;
                    self.stage = STAGE_EVASIONS;
                }
                STAGE_EVASIONS => {
                    if self.idx < self.list.len {
                        let m = self.list.pick_best(self.idx);
                        self.idx += 1;
                        return Some(m);
                    }
                    self.stage = STAGE_DONE;
                }
                _ => return None,
            }
        }
    }
}
