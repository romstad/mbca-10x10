//! Iterative deepening, aspiration windows and a fail-soft principal variation
//! search.

use crate::board::{Board, Undo};
use crate::eval::EvalState;
use crate::history::{self, Conts, Histories, NULL_CONT};
use crate::movegen::{generate, is_quiet, MoveList, GEN_ALL};
use crate::movepick::{capture_cell, mvv, MovePicker};
use crate::params::*;
use crate::see::{see_ge, see_value};
use crate::tt::*;
use crate::types::*;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const MATE: i32 = 32000;
pub const MATE_IN_MAX: i32 = MATE - MAX_PLY as i32;
pub const INFINITE: i32 = 32001;
pub const VALUE_NONE: i32 = 32002;
pub const DRAW: i32 = 0;

/// Stack slots below ply 0, so that looking back a few plies is always valid.
const OFF: usize = 8;
/// How often the clock and the stop flag are polled, in nodes. A power of two.
const CHECK_INTERVAL: u64 = 512;
/// Stack for helper threads. The same as the main search thread's: a node
/// holds a 256-entry move list, and extensions can carry the search to
/// `MAX_PLY`.
const HELPER_STACK: usize = 64 * 1024 * 1024;

/// How many late quiets keep their identity for the history malus.
const MAX_QUIETS_TRIED: usize = 64;
/// The same, for noisy moves. The capture stage is captures and all promoting
/// moves, so one advanced pawn can contribute eighteen.
const MAX_CAPTURES_TRIED: usize = 48;

/// The written prefix of an uninitialised move buffer.
///
/// # Safety
///
/// The first `n` entries of `buf` must have been written.
#[inline(always)]
unsafe fn init_prefix(buf: &[MaybeUninit<Move>], n: usize) -> &[Move] {
    debug_assert!(n <= buf.len());
    std::slice::from_raw_parts(buf.as_ptr().cast::<Move>(), n)
}

// -- late move reduction table ----------------------------------------------

const LMR_DEPTHS: usize = 64;
const LMR_MOVES: usize = 256;

type LmrTable = [[i32; LMR_MOVES]; LMR_DEPTHS];

static LMR: AtomicPtr<LmrTable> = AtomicPtr::new(std::ptr::null_mut());

/// Builds the reduction table.
pub fn init() {
    let mut t = Box::new([[0i32; LMR_MOVES]; LMR_DEPTHS]);
    for d in 1..LMR_DEPTHS {
        for m in 1..LMR_MOVES {
            let ld = (d as f64).ln().powf(LMR_D_POW as f64 / 100.0);
            let lm = (m as f64).ln().powf(LMR_M_POW as f64 / 100.0);
            t[d][m] = (LMR_BASE as f64 / 100.0 + ld * lm / (LMR_DIV as f64 / 100.0)) as i32;
        }
    }
    LMR.store(Box::leak(t), Ordering::Release);
}

#[inline(always)]
fn lmr(depth: i32, moves: i32) -> i32 {
    let d = (depth.max(0) as usize).min(LMR_DEPTHS - 1);
    let m = (moves.max(0) as usize).min(LMR_MOVES - 1);
    // Safe after `init()`, which every entry point calls before anything else.
    unsafe { (*LMR.load(Ordering::Acquire))[d][m] }
}

// -- limits, shared state, stack --------------------------------------------

#[derive(Clone, Default)]
pub struct Limits {
    pub time: [i64; 2],
    pub inc: [i64; 2],
    pub movestogo: Option<i64>,
    pub movetime: Option<i64>,
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub infinite: bool,
    pub ponder: bool,
    pub has_time: bool,
}

/// State shared between the searcher and the UCI thread.
pub struct Shared<'a> {
    pub tt: &'a Tt,
    pub stop: &'a AtomicBool,
    pub pondering: &'a AtomicBool,
    pub ponder_clock: Option<&'a Mutex<Instant>>,
    pub finished: AtomicBool,
    pub nodes: AtomicU64,
    pub start: Instant,
    pub soft_ms: u64,
    pub hard_ms: u64,
    pub limits: Limits,
    pub best: Mutex<(Move, i32, i32)>,
}

impl Shared<'_> {
    pub fn elapsed_ms(&self) -> u64 {
        if self.pondering.load(Ordering::Acquire) {
            return 0;
        }
        self.ponder_clock.map_or_else(
            || self.start.elapsed().as_millis() as u64,
            |clock| clock.lock().unwrap().elapsed().as_millis() as u64,
        )
    }
}

#[derive(Clone, Copy)]
pub struct Stack {
    pub killers: [Move; 2],
    pub current_move: Move,
    pub cont_idx: usize,
    pub static_eval: i32,
    /// The move a singular search is proving unique, and so must skip.
    pub excluded: Move,
    /// How many times a child of this node has failed high, which is evidence
    /// that the node is badly ordered and worth reducing harder.
    pub cutoff_cnt: i32,
    pub in_check: bool,
}

impl Default for Stack {
    fn default() -> Stack {
        Stack {
            killers: [Move::NONE; 2],
            current_move: Move::NONE,
            cont_idx: NULL_CONT,
            static_eval: VALUE_NONE,
            excluded: Move::NONE,
            cutoff_cnt: 0,
            in_check: false,
        }
    }
}

/// One finished principal variation of one iteration.
pub struct RootLine {
    pub mv: Move,
    pub score: i32,
    pub pv: Vec<Move>,
    pub sel_depth: usize,
}

pub struct Searcher {
    pub id: usize,
    pub board: Board,
    pub eval: EvalState,
    pub hist: Histories,
    ss: Vec<Stack>,
    undos: Vec<Undo>,
    pv: Vec<Vec<Move>>,
    pv_len: Vec<usize>,
    pub nodes: u64,
    /// How much of `nodes` has already been handed to the shared counter.
    shared_nodes: u64,
    sel_depth: usize,
    /// The depth of the iteration in progress, which bounds how far extensions
    /// may carry the search past it.
    root_depth: i32,
    pub best_move: Move,
    pub best_score: i32,
    /// The best move and score of the last completed iteration, which is
    /// what a helper thread contributes to the vote at the end of a search.
    pub final_move: Move,
    pub final_score: i32,
    pub completed_depth: i32,
    /// Suppresses `info` lines. Set by `searchtest`, which searches dozens of
    /// positions and only cares about the answers.
    pub quiet: bool,
    stopped: bool,
    root_moves: Vec<Move>,
    root_nodes: Vec<u64>,
    /// How many principal variations to search and report. Set per search by
    /// `go`, which gives every helper thread 1: a helper is worth more as extra
    /// depth on line one, through the shared table, than as a second opinion
    /// about line four.
    pub multipv: usize,
    /// Which line of the current iteration is being searched, 0-based.
    pv_idx: usize,
    /// The root moves earlier lines of this iteration have already claimed, and
    /// which the root move loop therefore skips. Always empty at `multipv` 1.
    root_excluded: Vec<Move>,
    /// The lines of the last completed iteration, best first.
    lines: Vec<RootLine>,
}

/// `cp N` or `mate N`, the way `info score` wants it.
pub fn score_to_uci(score: i32) -> String {
    if score.abs() >= MATE_IN_MAX {
        let plies = MATE - score.abs();
        let mates = (plies + 1) / 2;
        format!("mate {}", if score > 0 { mates } else { -mates })
    } else {
        format!("cp {score}")
    }
}

/// How much the opponent's last quiet move is credited, given the static
/// evaluation before it was answered and after.
///
/// Both arguments are side-to-move-relative, so they are from *opposite*
/// points of view: a large positive sum means both sides believe they stand
/// better, which is only possible if the move between them lost ground. The
/// leading minus turns that into a malus, and the offset is added after the
/// clamp so that a neutral swing still leaves a small positive tempo bonus.
#[inline(always)]
fn opp_swing_bonus(prev_eval: i32, static_eval: i32) -> i32 {
    (-OPP_HIST_MUL * (prev_eval + static_eval)).clamp(OPP_HIST_MIN, OPP_HIST_MAX) + OPP_HIST_OFF
}

#[inline(always)]
fn mate_in(ply: usize) -> i32 {
    MATE - ply as i32
}
#[inline(always)]
fn mated_in(ply: usize) -> i32 {
    -MATE + ply as i32
}

fn score_to_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_IN_MAX {
        s + ply as i32
    } else if s <= -MATE_IN_MAX {
        s - ply as i32
    } else {
        s
    }
}
fn score_from_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_IN_MAX {
        s - ply as i32
    } else if s <= -MATE_IN_MAX {
        s + ply as i32
    } else {
        s
    }
}

// -- time management --------------------------------------------------------

/// Splits the clock into a soft budget (checked between iterations) and a hard
/// one (checked inside the search).
pub fn time_limits(limits: &Limits, stm: Color, overhead: i64) -> (u64, u64) {
    if let Some(mt) = limits.movetime {
        let t = (mt - overhead).max(1) as u64;
        return (t, t);
    }
    if !limits.has_time {
        return (u64::MAX, u64::MAX);
    }
    let usable = (limits.time[stm.idx()] - overhead).max(1);
    let inc = limits.inc[stm.idx()];

    let (mut soft, mut hard) = match limits.movestogo {
        Some(mtg) if mtg > 0 => {
            let mtg = mtg.min(40);
            let soft = usable / mtg + inc * TM_INC_NUM as i64 / 100;
            (soft, (usable / 2).min(soft * TM_HARD_MULT as i64))
        }
        _ => {
            let soft = usable / TM_SOFT_DIV as i64 + inc * TM_INC_NUM as i64 / 100;
            (
                soft,
                (usable / TM_HARD_DIV as i64).min(soft * TM_HARD_MULT as i64),
            )
        }
    };
    soft = soft.clamp(1, (usable * 4 / 5).max(1));
    hard = hard.clamp(1, (usable * 9 / 10).max(1)).max(soft);
    (soft as u64, hard as u64)
}

// -- the searcher -----------------------------------------------------------

impl Searcher {
    pub fn new(id: usize) -> Searcher {
        Searcher {
            id,
            board: Board::startpos(),
            eval: EvalState::new(),
            hist: Histories::new(),
            ss: vec![Stack::default(); MAX_PLY + OFF + 8],
            undos: vec![Undo::default(); MAX_PLY + OFF + 8],
            pv: vec![vec![Move::NONE; MAX_PLY + 2]; MAX_PLY + 2],
            pv_len: vec![0; MAX_PLY + 2],
            nodes: 0,
            shared_nodes: 0,
            sel_depth: 0,
            root_depth: 0,
            best_move: Move::NONE,
            best_score: 0,
            final_move: Move::NONE,
            final_score: 0,
            completed_depth: 0,
            quiet: false,
            stopped: false,
            root_moves: Vec::new(),
            root_nodes: Vec::new(),
            multipv: 1,
            pv_idx: 0,
            root_excluded: Vec::new(),
            lines: Vec::new(),
        }
    }

    /// Clears everything a new game invalidates. The transposition table is the
    /// caller's business, since it is shared.
    pub fn clear(&mut self) {
        self.nodes = 0;
        self.shared_nodes = 0;
        self.completed_depth = 0;
        self.best_move = Move::NONE;
        self.hist.clear();
    }

    /// The continuation-history planes of the four preceding moves.
    ///
    /// `OFF` is 8, so `ply - 6` is in range even at the root.
    #[inline(always)]
    fn conts(&self, ply: usize) -> Conts {
        let p = ply + OFF;
        [
            self.ss[p - 1].cont_idx,
            self.ss[p - 2].cont_idx,
            self.ss[p - 4].cont_idx,
            self.ss[p - 6].cont_idx,
        ]
    }

    /// What the network says about this position, and nothing else.
    ///
    /// This is the only value the transposition table is allowed to cache. It
    /// is a function of piece placement and side to move alone, so it means the
    /// same thing at every node that reaches this position, which the adjusted
    /// evaluation does not: fifty-move damping depends on a halfmove clock the
    /// position key does not encode, and the correction is thread-local and
    /// keyed on the move that led here.
    fn evaluate_raw(&mut self) -> i32 {
        let raw = self.eval.evaluate(&self.board);
        debug_assert_eq!(
            raw,
            crate::eval::evaluate_scratch(&self.board),
            "incremental evaluation diverged from a full recomputation"
        );
        raw
    }

    /// The static evaluation as the search should believe it *here*: the raw
    /// score, damped by the fifty-move counter, then corrected by what the
    /// search has historically found this kind of position to be worth.
    fn adjust_eval(&self, raw: i32, ply: usize) -> i32 {
        // Damp the score as the fifty-move counter runs up, so the search
        // prefers a line that makes progress.
        let s = EVAL_FIFTY_SCALE;
        let v = raw * (s - self.board.halfmove as i32) / s;
        let prev = self.ss[ply + OFF - 1].cont_idx;
        let corrected = v + self.hist.correction(&self.board, prev);
        corrected.clamp(-MATE_IN_MAX + 1, MATE_IN_MAX - 1)
    }

    /// Both halves at once, for the sites that have no cached value to start
    /// from: the max-ply fallback in both searches.
    fn evaluate(&mut self, ply: usize) -> i32 {
        let raw = self.evaluate_raw();
        self.adjust_eval(raw, ply)
    }

    fn is_draw(&self, ply: usize) -> bool {
        if self.board.halfmove >= 100 {
            // Mate on the hundredth ply is mate, not a draw.
            if self.board.in_check() {
                let mut list = MoveList::new();
                generate::<GEN_ALL>(&self.board, &mut list);
                if list.len == 0 {
                    return false;
                }
            }
            return true;
        }
        self.board.insufficient_material() || self.board.has_repetition(ply)
    }

    fn check_stop(&mut self, sh: &Shared) -> bool {
        if self.stopped {
            return true;
        }
        if self.nodes % CHECK_INTERVAL == 0 {
            if self.nodes > self.shared_nodes {
                sh.nodes
                    .fetch_add(self.nodes - self.shared_nodes, Ordering::Relaxed);
                self.shared_nodes = self.nodes;
            }
            if sh.stop.load(Ordering::Relaxed) || sh.finished.load(Ordering::Relaxed) {
                self.stopped = true;
                return true;
            }
            // Only the reporting thread owns the clock.
            if self.id == 0 {
                let searched = self.completed_depth > 0;
                let over_time = searched
                    && !sh.limits.infinite
                    && !sh.pondering.load(Ordering::Acquire)
                    && sh.elapsed_ms() >= sh.hard_ms;
                let over_nodes = searched
                    && sh
                        .limits
                        .nodes
                        .is_some_and(|n| sh.nodes.load(Ordering::Relaxed) >= n);
                if over_time || over_nodes {
                    self.stopped = true;
                    return true;
                }
            }
        }
        false
    }

    /// The only place a move is made. Keeps `Board::hist`, the evaluation stack
    /// and the search stack in step.
    #[inline]
    fn do_move<const SAVE_CHECK_SQUARES: bool>(&mut self, m: Move, ply: usize) {
        let piece = self.board.piece_at(m.from());
        self.ss[ply + OFF].current_move = m;
        self.ss[ply + OFF].cont_idx = history::cont_index(piece, m.to());
        if SAVE_CHECK_SQUARES {
            self.board.make_move(m, &mut self.undos[ply + OFF]);
        } else {
            self.board
                .make_move_without_check_squares(m, &mut self.undos[ply + OFF]);
        }
        self.eval.push(&self.board, &self.undos[ply + OFF].dirty);
        self.nodes += 1;
    }

    #[inline]
    fn undo_move<const SAVE_CHECK_SQUARES: bool>(&mut self, m: Move, ply: usize) {
        self.eval.pop();
        // Borrowed, not copied: `Undo` is some 200 bytes, of which
        // `check_squares` alone is 128, and `board` and `undos` are disjoint
        // fields.
        let undo = &self.undos[ply + OFF];
        if SAVE_CHECK_SQUARES {
            self.board.unmake_move(m, undo);
        } else {
            self.board.unmake_move_without_check_squares(m, undo);
        }
    }

    fn update_pv(&mut self, ply: usize, m: Move) {
        self.pv[ply][0] = m;
        let child = self.pv_len[ply + 1];
        for i in 0..child {
            self.pv[ply][i + 1] = self.pv[ply + 1][i];
        }
        self.pv_len[ply] = child + 1;
    }

    // -- history updates ----------------------------------------------------

    /// Rewards or punishes a quiet move in every table that scores quiets, so
    /// that the picker and the reductions see one consistent story.
    ///
    /// `m` must be pseudo-legal in the current position: `piece_at(m.from())`
    /// indexes the mailbox unchecked, and `Piece::idx()` then indexes tables of
    /// sixteen planes, where `Piece::NONE` is 16.
    fn update_quiet_histories(&mut self, m: Move, bonus: i32, ply: usize) {
        debug_assert!(
            self.board.is_pseudo_legal(m),
            "quiet history on an illegal move"
        );
        let stm = self.board.stm;
        self.hist.update_main(stm, m, bonus);
        let piece = self.board.piece_at(m.from());
        self.hist
            .update_pawn(self.board.pawn_key, piece, m.to(), bonus);
        for (k, &c) in self.conts(ply).iter().enumerate() {
            if c == NULL_CONT {
                continue;
            }
            // The two most recent plies are the informative ones; the older
            // pair gets half weight.
            let scaled = if k < 2 { bonus } else { bonus / 2 };
            self.hist.update_cont(c, piece, m.to(), scaled);
        }
    }

    /// Credits the move that caused a fail-high and debits everything tried
    /// before it.
    fn update_histories(
        &mut self,
        best_move: Move,
        depth: i32,
        ply: usize,
        killers: [Move; 2],
        quiets_tried: &[Move],
        captures_tried: &[Move],
    ) {
        let bonus = (HIST_BONUS_SLOPE * depth - HIST_BONUS_OFF).clamp(0, 1900);
        let malus = (HIST_MALUS_SLOPE * depth - HIST_MALUS_OFF).clamp(0, 1400);

        if is_quiet(&self.board, best_move) {
            self.update_quiet_histories(best_move, bonus, ply);
            self.ss[ply + OFF].killers = if killers[0] == best_move {
                killers
            } else {
                [best_move, killers[0]]
            };
            let prev = self.ss[ply + OFF - 1].cont_idx;
            if prev != NULL_CONT {
                self.hist.set_counter_move(prev, best_move);
            }
            for &q in quiets_tried {
                if q != best_move {
                    self.update_quiet_histories(q, -malus, ply);
                }
            }
        } else {
            let (p, to, victim) = capture_cell(&self.board, best_move);
            self.hist.update_capture(p, to, victim, bonus);
        }
        for &c in captures_tried {
            if c != best_move {
                let (p, to, victim) = capture_cell(&self.board, c);
                self.hist.update_capture(p, to, victim, -malus);
            }
        }
    }

    /// The most valuable type the side to move could still promote to, or
    /// `None` when the inventory is empty and no promotion is legal at all.
    fn best_promo(&self) -> Option<PieceType> {
        let inv = self.board.promo_inventory(self.board.stm);
        PROMO_TYPES
            .iter()
            .copied()
            .filter(|pt| inv & (1u8 << pt.idx()) != 0)
            .max_by_key(|&pt| see_value(pt))
    }

    // -- quiescence ---------------------------------------------------------

    fn qsearch<const PV: bool>(
        &mut self,
        sh: &Shared,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> i32 {
        if PV {
            self.pv_len[ply] = 0;
        }
        if self.check_stop(sh) {
            return DRAW;
        }
        self.sel_depth = self.sel_depth.max(ply);

        if ply >= MAX_PLY {
            return self.evaluate(ply);
        }
        if self.is_draw(ply) {
            return DRAW;
        }

        let in_check = self.board.in_check();
        let hit = sh.tt.probe(self.board.key);
        let tt_move = hit.map_or(Move::NONE, |h| h.mv);
        if !PV {
            if let Some(h) = hit {
                let s = score_from_tt(h.score, ply);
                let usable = match h.bound {
                    BOUND_EXACT => true,
                    BOUND_LOWER => s >= beta,
                    BOUND_UPPER => s <= alpha,
                    _ => false,
                };
                if usable {
                    return s;
                }
            }
        }

        // Stand pat. In check there is no standing still: every evasion must be
        // examined, so the floor starts at -INFINITE and the picker generates
        // the whole move list rather than just the captures.
        let raw_nnue;
        let mut best = if in_check {
            raw_nnue = VALUE_NONE;
            -INFINITE
        } else {
            let raw = match hit {
                Some(h) if h.eval != VALUE_NONE && h.eval.abs() < MATE_IN_MAX => h.eval,
                _ => self.evaluate_raw(),
            };
            raw_nnue = raw;
            let static_eval = self.adjust_eval(raw, ply);
            // A hash score is a better estimate of the node than the static
            // evaluation, when the bound points the right way.
            let mut ev = static_eval;
            if let Some(h) = hit {
                let s = score_from_tt(h.score, ply);
                let better = match h.bound {
                    BOUND_EXACT => true,
                    BOUND_LOWER => s > ev,
                    BOUND_UPPER => s < ev,
                    _ => false,
                };
                if better {
                    ev = s;
                }
            }
            if ev >= beta {
                if hit.is_none() {
                    sh.tt.store(
                        self.board.key,
                        Move::NONE,
                        score_to_tt(ev, ply),
                        raw_nnue,
                        0,
                        BOUND_LOWER,
                        PV,
                    );
                }
                return ev;
            }
            alpha = alpha.max(ev);
            ev
        };

        self.ss[ply + OFF].in_check = in_check;
        let conts = self.conts(ply);
        let mut mp = MovePicker::new_qsearch(tt_move, in_check);
        let mut best_move = Move::NONE;
        let mut moves = 0;
        let futility = best + QS_FUTILITY;

        while let Some(m) = mp.next(&self.board, &self.hist, &conts, true) {
            moves += 1;
            if !in_check && best > -MATE_IN_MAX {
                // Delta pruning: even winning the victim outright would not
                // reach alpha, so the move cannot matter.
                if !(m.is_promotion() && QS_PROMO_DELTA == 0) {
                    let gain = mvv(&self.board, m);
                    if futility + gain <= alpha {
                        best = best.max(futility + gain);
                        continue;
                    }
                }
                // A losing capture cannot raise alpha in quiescence often
                // enough to pay for its subtree.
                if !see_ge(&self.board, m, -QS_SEE) {
                    continue;
                }
            }
            self.do_move::<false>(m, ply);
            sh.tt.prefetch(self.board.key);
            let score = -self.qsearch::<PV>(sh, -beta, -alpha, ply + 1);
            self.undo_move::<false>(m, ply);
            if self.stopped {
                return DRAW;
            }
            if score > best {
                best = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    if PV {
                        self.update_pv(ply, m);
                    }
                    if score >= beta {
                        break;
                    }
                }
            }
        }

        if in_check && moves == 0 {
            return mated_in(ply);
        }

        let bound = if best >= beta {
            BOUND_LOWER
        } else {
            BOUND_UPPER
        };
        sh.tt.store(
            self.board.key,
            best_move,
            score_to_tt(best, ply),
            raw_nnue,
            0,
            bound,
            PV,
        );
        best
    }

    // -- main search --------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn search<const PV: bool>(
        &mut self,
        sh: &Shared,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        cut_node: bool,
        ply: usize,
    ) -> i32 {
        if depth <= 0 {
            return self.qsearch::<PV>(sh, alpha, beta, ply);
        }
        if PV {
            self.pv_len[ply] = 0;
        }
        if self.check_stop(sh) {
            return DRAW;
        }
        let is_root = ply == 0;

        if !is_root {
            if self.is_draw(ply) {
                return DRAW;
            }
            if ply >= MAX_PLY {
                return self.evaluate(ply);
            }
            // Mate distance pruning: no line from here can beat a mate already
            // found closer to the root.
            alpha = alpha.max(mated_in(ply));
            beta = beta.min(mate_in(ply + 1));
            if alpha >= beta {
                return alpha;
            }
        }

        // A singular search re-enters this node with one move removed. It must
        // not see the table entry it is testing, and must not write one back.
        let excluded = self.ss[ply + OFF].excluded;
        let hit = if excluded.is_none() {
            sh.tt.probe(self.board.key)
        } else {
            None
        };
        let mut tt_move = if is_root { self.best_move } else { Move::NONE };
        let mut tt_score = VALUE_NONE;
        let mut tt_depth = -1;
        let mut tt_bound = BOUND_NONE;
        let mut tt_pv = PV;
        if let Some(h) = hit {
            if !is_root || tt_move.is_none() {
                tt_move = h.mv;
            }
            tt_score = score_from_tt(h.score, ply);
            tt_depth = h.depth;
            tt_bound = h.bound;
            tt_pv = tt_pv || h.was_pv;
            if !PV && !is_root && h.depth >= depth {
                let usable = match h.bound {
                    BOUND_EXACT => true,
                    BOUND_LOWER => tt_score >= beta,
                    BOUND_UPPER => tt_score <= alpha,
                    _ => false,
                };
                if usable {
                    // A cutoff on a hash hit still deserves a history nudge,
                    // but `is_pseudo_legal` has to run first.
                    if tt_score >= beta
                        && !h.mv.is_none()
                        && self.board.is_pseudo_legal(h.mv)
                        && is_quiet(&self.board, h.mv)
                    {
                        let bonus = (HIST_BONUS_SLOPE * depth - HIST_BONUS_OFF).clamp(0, 1900);
                        self.update_quiet_histories(h.mv, bonus, ply);
                    }
                    // Near the fifty-move boundary a transposition is not
                    // really the same position: the draw clock differs.
                    if self.board.halfmove < 90 {
                        return tt_score;
                    }
                }
            }
        }

        let in_check = self.board.in_check();
        self.ss[ply + OFF].in_check = in_check;
        self.ss[ply + OFF + 1].killers = [Move::NONE; 2];
        self.ss[ply + OFF + 1].cutoff_cnt = 0;

        // -- static evaluation ----------------------------------------------

        let raw_nnue;
        let static_eval;
        let mut eval;
        if in_check {
            raw_nnue = VALUE_NONE;
            static_eval = VALUE_NONE;
            eval = VALUE_NONE;
            self.ss[ply + OFF].static_eval = VALUE_NONE;
        } else if !excluded.is_none() {
            // The singular search is the same position and the same path: the
            // stack's adjusted value is already right, and adjusting it again
            // would apply the correction twice. Re-storing it would overwrite
            // the entry being tested, so no raw value is invented here either.
            raw_nnue = VALUE_NONE;
            static_eval = self.ss[ply + OFF].static_eval;
            eval = static_eval;
        } else {
            let raw = match hit {
                Some(h) if h.eval != VALUE_NONE && h.eval.abs() < MATE_IN_MAX => h.eval,
                _ => self.evaluate_raw(),
            };
            raw_nnue = raw;
            static_eval = self.adjust_eval(raw, ply);
            eval = static_eval;
            if tt_score != VALUE_NONE {
                let better = match tt_bound {
                    BOUND_EXACT => true,
                    BOUND_LOWER => tt_score > eval,
                    BOUND_UPPER => tt_score < eval,
                    _ => false,
                };
                if better {
                    eval = tt_score;
                }
            }
            self.ss[ply + OFF].static_eval = static_eval;
            if hit.is_none() {
                sh.tt.store(
                    self.board.key,
                    Move::NONE,
                    VALUE_NONE,
                    raw_nnue,
                    -1,
                    BOUND_NONE,
                    tt_pv,
                );
            }
        }

        // -- the opponent's last quiet move, judged by the eval swing --------

        if OPP_HIST_MUL != 0 && !in_check && excluded.is_none() && ply >= 1 {
            let pm = self.ss[ply + OFF - 1].current_move;
            let prev_eval = self.ss[ply + OFF - 1].static_eval;
            if !pm.is_none()
                && pm != Move::NULL
                && !pm.is_promotion()
                && prev_eval != VALUE_NONE
                && self.undos[ply + OFF - 1].captured.is_none()
            {
                let bonus = opp_swing_bonus(prev_eval, static_eval);
                self.hist.update_main(self.board.stm.flip(), pm, bonus);
            }
        }

        // Are we doing better than we were two plies ago? A node that is improving
        // can afford to prune less and reduce less.
        let improving = if in_check {
            false
        } else {
            let prev = self.ss[ply + OFF - 2].static_eval;
            if prev != VALUE_NONE {
                static_eval > prev
            } else {
                let prev4 = self.ss[ply + OFF - 4].static_eval;
                prev4 == VALUE_NONE || static_eval > prev4
            }
        };

        // -- whole-node pruning ---------------------------------------------
        if !PV && !in_check && excluded.is_none() && beta.abs() < MATE_IN_MAX {
            // Reverse futility pruning.
            if depth <= RFP_DEPTH && eval - RFP_MARGIN * (depth - improving as i32) >= beta {
                return beta + (eval - beta) / 3;
            }

            // Razoring.
            if depth <= RAZOR_DEPTH && eval + RAZOR_BASE + RAZOR_SLOPE * depth < alpha {
                let v = self.qsearch::<false>(sh, alpha - 1, alpha, ply);
                if v < alpha {
                    return v;
                }
            }

            // Null move pruning.
            if depth >= NMP_MIN_DEPTH
                && eval >= beta
                && self.ss[ply + OFF].static_eval + NMP_VERIFY_SLOPE * depth >= beta
                && !self.board.only_pawns(self.board.stm)
                && self.ss[ply + OFF - 1].current_move != Move::NULL
            {
                let r = NMP_BASE
                    + depth / NMP_DEPTH_DIV
                    + ((eval - beta) / NMP_EVAL_DIV).min(NMP_EVAL_MAX);
                self.ss[ply + OFF].current_move = Move::NULL;
                self.ss[ply + OFF].cont_idx = NULL_CONT;
                self.board.make_null(&mut self.undos[ply + OFF]);
                self.eval.push_null();
                let score =
                    -self.search::<false>(sh, -beta, -beta + 1, depth - r, !cut_node, ply + 1);
                self.eval.pop();
                self.board.unmake_null(&self.undos[ply + OFF]);
                if self.stopped {
                    return DRAW;
                }
                if score >= beta {
                    // A mate score proved by not moving is not a mate.
                    return if score >= MATE_IN_MAX { beta } else { score };
                }
            }
        }

        // ProbCut
        if !PV
            && !in_check
            && excluded.is_none()
            && depth >= PROBCUT_DEPTH
            && beta.abs() < MATE_IN_MAX
            && eval != VALUE_NONE
        {
            let pc_beta = beta + PROBCUT_MARGIN - PROBCUT_IMPROVING * improving as i32;
            let deep_enough = tt_depth >= depth - 3 && tt_score != VALUE_NONE;
            if !(deep_enough && tt_score < pc_beta) {
                let conts_pc = self.conts(ply);
                let mut mp = MovePicker::new_probcut(tt_move, pc_beta - eval);
                while let Some(m) = mp.next(&self.board, &self.hist, &conts_pc, true) {
                    if m == excluded {
                        continue;
                    }
                    self.do_move::<true>(m, ply);
                    let mut score = -self.qsearch::<false>(sh, -pc_beta, -pc_beta + 1, ply + 1);
                    if score >= pc_beta {
                        score = -self.search::<false>(
                            sh,
                            -pc_beta,
                            -pc_beta + 1,
                            depth - PROBCUT_RED,
                            !cut_node,
                            ply + 1,
                        );
                    }
                    self.undo_move::<true>(m, ply);
                    if self.stopped {
                        return DRAW;
                    }
                    if score >= pc_beta {
                        sh.tt.store(
                            self.board.key,
                            m,
                            score_to_tt(score, ply),
                            raw_nnue,
                            depth - (PROBCUT_RED - 1),
                            BOUND_LOWER,
                            tt_pv,
                        );
                        return score;
                    }
                }
            }
        }

        // Internal iterative reduction.
        if depth >= IIR_DEPTH && tt_move.is_none() && !is_root {
            depth -= 1 + PV as i32;
            if depth <= 0 {
                return self.qsearch::<PV>(sh, alpha, beta, ply);
            }
        }

        self.board.update_check_squares();

        let orig_alpha = alpha;
        let conts = self.conts(ply);
        let killers = self.ss[ply + OFF].killers;
        let counter = {
            let prev = self.ss[ply + OFF - 1].cont_idx;
            if prev == NULL_CONT {
                Move::NONE
            } else {
                self.hist.counter_move(prev)
            }
        };
        let mut mp = MovePicker::new(tt_move, killers, counter, in_check);

        let mut best = -INFINITE;
        let mut best_move = Move::NONE;
        let mut moves = 0;
        let mut skip_quiets = false;
        let mut quiets_tried: [MaybeUninit<Move>; MAX_QUIETS_TRIED] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut n_quiets = 0usize;
        let mut captures_tried: [MaybeUninit<Move>; MAX_CAPTURES_TRIED] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut n_captures = 0usize;
        // Resolved lazily: it costs six popcounts and most nodes never ask.
        let mut best_promo: Option<Option<PieceType>> = None;
        // MultiPV: how many root moves the earlier lines have already claimed.
        // Zero at every node but the root, and zero at the root unless MultiPV
        // is on, so the loop below pays one predictable register test for it and
        // never dereferences the vector in the default search.
        let n_skip = if is_root { self.root_excluded.len() } else { 0 };

        while let Some(m) = mp.next(&self.board, &self.hist, &conts, skip_quiets) {
            if m == excluded {
                continue;
            }
            if n_skip != 0 && self.root_excluded.contains(&m) {
                continue;
            }
            moves += 1;
            let quiet = is_quiet(&self.board, m);
            let gives_check = self.board.gives_check(m);
            let before = self.nodes;

            // The same statistics the picker sorted by, which the pruning and
            // the reduction below both consult. Only the two most recent
            // continuation planes: the older pair is too noisy to prune on.
            let piece = self.board.piece_at(m.from());
            // The victim type of a noisy move, hoisted because capture futility
            // needs the same one. `King` stands for "captured nothing", and
            // `see_value(King)` is 0, so a non-capturing promotion contributes
            // nothing to the margin.
            let mut victim = PieceType::King;
            let hist_score = if quiet {
                let mut s = self.hist.main_score(self.board.stm, m)
                    + self.hist.pawn_score(self.board.pawn_key, piece, m.to());
                for &c in conts.iter().take(2) {
                    if c != NULL_CONT {
                        s += self.hist.cont_score(c, piece, m.to());
                    }
                }
                s
            } else {
                let (p, to, v) = capture_cell(&self.board, m);
                victim = v;
                self.hist.capture_score(p, to, v)
            };

            // -- move-level pruning -----------------------------------------
            if !is_root && best > -MATE_IN_MAX {
                // The depth this move will actually be searched at, which is
                // what the margins have to be measured against.
                let lmr_depth = (depth - lmr(depth, moves)).max(0);
                if quiet {
                    // Late move pruning
                    if !skip_quiets && moves >= (LMP_BASE + depth * depth) / (2 - improving as i32)
                    {
                        skip_quiets = true;
                    }
                    // Futility: this quiet cannot lift a losing position to
                    // alpha, and neither can any quiet after it.
                    if !in_check
                        && lmr_depth <= FP_MAX_DEPTH
                        && eval != VALUE_NONE
                        && eval.abs() < MATE_IN_MAX
                        && eval + FP_BASE + FP_SLOPE * lmr_depth <= alpha
                    {
                        skip_quiets = true;
                        continue;
                    }
                    // History pruning: a move this consistently bad is not
                    // worth a subtree.
                    if lmr_depth <= HP_MAX_DEPTH && hist_score < -HP_SLOPE * depth {
                        continue;
                    }
                    if !see_ge(&self.board, m, -SEE_QUIET * lmr_depth * lmr_depth) {
                        continue;
                    }
                } else {
                    // -- capture futility, in two variants ------------------

                    let futile =
                        !in_check && !gives_check && !m.is_promotion() && eval.abs() < MATE_IN_MAX;

                    // (a) Everything left is a losing capture, ordered worst
                    // last, and even a free move would not reach alpha. Stop.
                    if futile && depth < BNFP_DEPTH && mp.in_bad_captures() {
                        let fv = eval + BNFP_BASE + BNFP_SLOPE * depth;
                        if fv <= alpha {
                            best = best.max(fv);
                            break;
                        }
                    }

                    // (b) This capture cannot lift the node to alpha even if it
                    // wins its victim outright.
                    if !PV && futile && m != tt_move && lmr_depth < CFP_DEPTH {
                        let fv = eval
                            + CFP_BASE
                            + CFP_SLOPE * lmr_depth
                            + see_value(victim)
                            + hist_score / CFP_HIST_DIV;
                        if fv <= alpha {
                            best = best.max(fv);
                            continue;
                        }
                    }

                    if depth <= SEE_NOISY_DEPTH && !see_ge(&self.board, m, -SEE_NOISY * depth) {
                        continue;
                    }
                }

                // Underpromotion pruning, off by default.
                if UNDERPROMO_PRUNE_DEPTH > 0
                    && depth <= UNDERPROMO_PRUNE_DEPTH
                    && m.is_promotion()
                    && self.board.piece_at(m.to()).is_none()
                    && !gives_check
                {
                    let best_pt = *best_promo.get_or_insert_with(|| self.best_promo());
                    if best_pt.is_some_and(|pt| pt != m.promo()) {
                        continue;
                    }
                }
            }

            // -- extensions --------------------------------------------------
            //
            // Bounded by `root_depth * 2`, so that a chain of extensions cannot
            // run the search away from the iteration it belongs to.
            let mut extension = 0;
            if !is_root && ply < (self.root_depth * 2) as usize {
                if depth >= SE_MIN_DEPTH
                    && m == tt_move
                    && excluded.is_none()
                    && tt_depth >= depth - 3
                    && (tt_bound & BOUND_LOWER) != 0
                    && tt_score.abs() < MATE_IN_MAX
                {
                    // Singular extension: search everything *except* the hash
                    // move against a lowered window. If nothing else comes
                    // close, the hash move is the only move and is worth a ply.
                    let singular_beta = tt_score - depth * SE_BETA_SLOPE;
                    let singular_depth = (depth - 1) / 2;
                    self.ss[ply + OFF].excluded = m;
                    let score = self.search::<false>(
                        sh,
                        singular_beta - 1,
                        singular_beta,
                        singular_depth,
                        cut_node,
                        ply,
                    );
                    self.ss[ply + OFF].excluded = Move::NONE;
                    if self.stopped {
                        return DRAW;
                    }
                    if score < singular_beta {
                        extension = 1;
                        if !PV && score < singular_beta - SE_DOUBLE_MARGIN {
                            extension = 2;
                        }
                    } else if singular_beta >= beta {
                        // Multi-cut: several moves beat beta, so this node is a
                        // fail-high without searching any of them properly.
                        return singular_beta;
                    } else if tt_score >= beta {
                        extension = -2;
                    } else if cut_node {
                        extension = -1;
                    }
                } else if gives_check && depth > CHECK_EXT_DEPTH {
                    extension = 1;
                }
            }
            let new_depth = depth - 1 + extension;

            self.do_move::<true>(m, ply);
            sh.tt.prefetch(self.board.key);
            debug_assert_eq!(
                gives_check,
                self.board.in_check(),
                "gives_check disagreed with the position after {}",
                move_to_string(m)
            );

            // -- late move reductions, then principal variation search ------
            //
            // A late move is searched shallow against a null window first. If
            // it unexpectedly beats alpha, it earns the full depth; if it beats
            // alpha in a PV node, it earns the full window too.
            let mut score;
            let full_depth_search;
            if depth >= 2 && moves > 1 + (is_root as i32) * 2 && (quiet || !tt_pv) {
                let mut r = lmr(depth, moves);
                r += !PV as i32;
                r += cut_node as i32;
                r -= tt_pv as i32;
                r -= improving as i32;
                r -= gives_check as i32;
                let div = if quiet { LMR_HIST_DIV } else { LMR_CAPT_DIV };
                r -= (hist_score / div).clamp(-2, 2);
                // A node whose children keep failing high is badly ordered, so
                // trust its ordering less.
                if self.ss[ply + OFF + 1].cutoff_cnt > 3 {
                    r += 1;
                }
                let d = (new_depth - r).clamp(1, new_depth.max(1));
                score = -self.search::<false>(sh, -alpha - 1, -alpha, d, true, ply + 1);
                full_depth_search = score > alpha && d < new_depth;
            } else {
                full_depth_search = !PV || moves > 1;
                score = alpha + 1;
            }

            if full_depth_search {
                score =
                    -self.search::<false>(sh, -alpha - 1, -alpha, new_depth, !cut_node, ply + 1);
            }
            if PV && (moves == 1 || score > alpha) {
                score = -self.search::<PV>(sh, -beta, -alpha, new_depth, false, ply + 1);
            }
            self.undo_move::<true>(m, ply);

            if self.stopped {
                return DRAW;
            }
            if is_root {
                if let Some(idx) = self.root_moves.iter().position(|&r| r == m) {
                    self.root_nodes[idx] += self.nodes - before;
                }
            }

            if score > best {
                best = score;
                best_move = m;
                if score > alpha {
                    alpha = score;
                    if PV {
                        self.update_pv(ply, m);
                    }
                    if is_root {
                        self.best_move = m;
                        self.best_score = score;
                    }
                    if score >= beta {
                        self.ss[ply + OFF].cutoff_cnt += 1;
                        break;
                    }
                }
            }
            // Recorded after the cutoff check: the move that caused it is
            // credited separately and must not also collect the malus.
            if quiet {
                if n_quiets < MAX_QUIETS_TRIED {
                    quiets_tried[n_quiets].write(m);
                    n_quiets += 1;
                }
            } else if n_captures < MAX_CAPTURES_TRIED {
                captures_tried[n_captures].write(m);
                n_captures += 1;
            }
        }

        if moves == 0 {
            // Under exclusion this is not a terminal node at all: it is the
            // node minus one move, and `alpha` is the answer the singular
            // search is asking for.
            return if !excluded.is_none() {
                alpha
            } else if in_check {
                mated_in(ply)
            } else {
                DRAW
            };
        }
        if !best_move.is_none() && best >= beta {
            self.update_histories(
                best_move,
                depth,
                ply,
                killers,
                // Safe: exactly the first `n_quiets` and `n_captures` entries
                // were written above, and `MaybeUninit<Move>` is `Move`'s
                // layout.
                unsafe { init_prefix(&quiets_tried, n_quiets) },
                unsafe { init_prefix(&captures_tried, n_captures) },
            );
        }

        // Nothing is written under exclusion: the entry being tested is the one
        // that would be overwritten, and its depth and bound are what made the
        // singular search worth running.
        if excluded.is_none() && !(is_root && self.pv_idx > 0) {
            let bound = if best >= beta {
                BOUND_LOWER
            } else if PV && !best_move.is_none() && best > orig_alpha {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            sh.tt.store(
                self.board.key,
                best_move,
                score_to_tt(best, ply),
                raw_nnue,
                depth,
                bound,
                tt_pv,
            );

            // Correction history: Learn the difference between what the
            // evaluation said and what the search found, keyed on the pawn
            // structure, the non-pawn material and the move that led here.
            if !in_check
                && static_eval != VALUE_NONE
                && (best_move.is_none() || is_quiet(&self.board, best_move))
                && !(best >= beta && best <= static_eval)
                && !(best_move.is_none() && best >= static_eval)
            {
                let prev = self.ss[ply + OFF - 1].cont_idx;
                self.hist
                    .update_correction(&self.board, prev, best - static_eval, depth);
            }
        }
        best
    }

    pub fn iterate(&mut self, sh: &Shared) {
        self.nodes = 0;
        self.shared_nodes = 0;
        self.sel_depth = 0;
        self.stopped = false;
        self.completed_depth = 0;
        self.best_move = Move::NONE;
        self.best_score = 0;
        self.final_move = Move::NONE;
        self.final_score = 0;
        self.pv_idx = 0;
        self.root_excluded.clear();
        self.lines.clear();
        self.eval.reset(&self.board);
        for s in self.ss.iter_mut() {
            *s = Stack::default();
        }

        let mut list = MoveList::new();
        generate::<GEN_ALL>(&self.board, &mut list);
        self.root_moves = (0..list.len).map(|i| list.get(i)).collect();
        self.root_nodes = vec![0; self.root_moves.len()];
        if self.root_moves.is_empty() {
            return;
        }
        self.best_move = self.root_moves[0];

        let max_depth = sh.limits.depth.unwrap_or(MAX_PLY as i32 - 1);
        // More lines than there are legal moves would leave the root with
        // nothing in it, and a root with no moves reads as mate or stalemate.
        let mpv = self.multipv.clamp(1, self.root_moves.len());
        let mut stability = 0usize;
        let mut last_best = Move::NONE;
        // Nodes spent on the lines past the first. The node-fraction scaling
        // below asks what share of the tree the *best* move took, and those
        // lines are not part of that tree. Stays zero at MultiPV 1.
        let mut extra_nodes = 0u64;
        // The iteration in progress, published to `self.lines` only once every
        // line of it has finished.
        let mut cur: Vec<RootLine> = Vec::new();

        for depth in 1..=max_depth {
            self.root_depth = depth;
            self.root_excluded.clear();
            cur.clear();

            for pv_idx in 0..mpv {
                self.pv_idx = pv_idx;
                self.sel_depth = 0;
                if mpv > 1 {
                    // The root takes its hash move from `best_move`, so this is
                    // how line k gets its own answer from the previous
                    // iteration tried first. Guarded, because at MultiPV 1
                    // `best_move` already carries exactly that and has to keep
                    // evolving across the aspiration re-searches below.
                    self.best_move = self.lines.get(pv_idx).map_or(Move::NONE, |l| l.mv);
                }
                // Aspiration windows: re-search a narrow window around *this
                // line's* previous score, widening on each failure.
                let prev_score = self.lines.get(pv_idx).map_or(0, |l| l.score);
                let mut delta = ASPIRATION_DELTA + prev_score * prev_score / 12000;
                let (mut alpha, mut beta) = if depth >= ASP_MIN_DEPTH {
                    (
                        (prev_score - delta).max(-INFINITE),
                        (prev_score + delta).min(INFINITE),
                    )
                } else {
                    (-INFINITE, INFINITE)
                };
                let before = self.nodes;

                // On a fail high the position is better than believed and the
                // move is usually obvious, so the re-search is allowed to give
                // up a ply; on a fail low it is not, because a refutation has
                // to be found.
                let mut fail_depth = depth;
                let score = loop {
                    let s = self.search::<true>(sh, alpha, beta, fail_depth.max(1), false, 0);
                    if self.stopped {
                        break s;
                    }
                    if s <= alpha {
                        beta = (alpha + beta) / 2;
                        alpha = (s - delta).max(-INFINITE);
                        fail_depth = depth;
                    } else if s >= beta {
                        beta = (s + delta).min(INFINITE);
                        fail_depth = (fail_depth - 1).max(depth - 2).max(1);
                    } else {
                        break s;
                    }
                    delta += delta / 3;
                };

                if pv_idx > 0 {
                    extra_nodes += self.nodes - before;
                }
                if self.stopped {
                    break;
                }
                // Snapshot the variation now: entering the root again zeroes
                // `pv_len[0]`, so the next line would otherwise destroy this one.
                cur.push(RootLine {
                    mv: self.best_move,
                    score,
                    pv: self.pv(),
                    sel_depth: self.sel_depth,
                });
                self.root_excluded.push(self.best_move);
                if mpv > 1 {
                    // A later line can outscore an earlier one: the windows are
                    // seeded per line and the table moves underneath them.
                    // Stable, so equal scores keep the order they were found in.
                    cur.sort_by_key(|l| std::cmp::Reverse(l.score));
                    self.best_move = cur[0].mv;
                    self.best_score = cur[0].score;
                }
            }
            self.pv_idx = 0;

            if self.stopped {
                // `bestmove` is line one's, never whichever line the stop
                // happened to land in. With nothing finished this leaves the
                // partial line-one result alone, which is what a single-PV
                // search has always played.
                if let Some(l) = cur.first() {
                    self.best_move = l.mv;
                    self.best_score = l.score;
                }
                break;
            }
            std::mem::swap(&mut self.lines, &mut cur);
            let score = self.lines[0].score;

            self.completed_depth = depth;
            self.final_move = self.best_move;
            self.final_score = score;

            stability = if self.best_move == last_best {
                (stability + 1).min(7)
            } else {
                0
            };
            last_best = self.best_move;

            if self.id == 0 {
                if !self.quiet {
                    self.report(sh, depth);
                }
                *sh.best.lock().unwrap() = (self.best_move, score, depth);
            }

            if sh.limits.infinite || sh.pondering.load(Ordering::Relaxed) {
                continue;
            }
            // A single legal move needs no more thought.
            if self.root_moves.len() == 1 && depth >= TM_MIN_DEPTH {
                break;
            }
            // A forced mate found within the remaining depth is the answer.
            if score.abs() >= MATE_IN_MAX && depth >= MATE - score.abs() + 4 {
                break;
            }

            if sh.limits.has_time || sh.limits.movetime.is_some() {
                const FACTORS: [f64; 8] = [2.30, 1.55, 1.28, 1.15, 1.05, 0.98, 0.93, 0.88];
                let mut soft =
                    sh.soft_ms as f64 * FACTORS[stability] * TM_STAB_SCALE as f64 / 100.0;
                // Spend less when the best move is taking most of the tree
                // anyway: there is little left to discover.
                if let Some(idx) = self.root_moves.iter().position(|&r| r == self.best_move) {
                    // Lines two and up are not part of the best move's tree, so
                    // they come out of the denominator. `extra_nodes` is zero at
                    // MultiPV 1, leaving the division it has always been.
                    let total = self.nodes - extra_nodes;
                    if total > 0 {
                        let frac = self.root_nodes[idx] as f64 / total as f64;
                        let base = TM_NODE_BASE as f64 / 100.0;
                        let scale = (base - frac * TM_NODE_SLOPE as f64 / 100.0).clamp(0.6, base);
                        soft *= scale;
                    }
                }
                if depth >= TM_MIN_DEPTH && sh.elapsed_ms() as f64 >= soft {
                    break;
                }
            }
            self.root_nodes = vec![0; self.root_moves.len()];
        }
    }

    /// The principal variation of the last completed iteration.
    pub fn pv(&self) -> Vec<Move> {
        (0..self.pv_len[0]).map(|i| self.pv[0][i]).collect()
    }

    /// The lines of the last *completed* iteration, best first.
    pub fn lines(&self) -> &[RootLine] {
        &self.lines
    }

    /// One `info` line per principal variation of the last completed iteration,
    /// best first.
    fn report(&self, sh: &Shared, depth: i32) {
        let elapsed = sh.elapsed_ms();
        let nodes = self.nodes.max(sh.nodes.load(Ordering::Relaxed));
        let nps = nodes * 1000 / elapsed.max(1);
        let hashfull = sh.tt.hashfull();
        for (i, line) in self.lines.iter().enumerate() {
            let score_str = score_to_uci(line.score);
            let mut pv = String::new();
            for &m in line.pv.iter() {
                if !pv.is_empty() {
                    pv.push(' ');
                }
                pv.push_str(&move_to_string(m));
            }
            if pv.is_empty() {
                pv = move_to_string(line.mv);
            }
            let multipv = if self.multipv > 1 {
                format!("multipv {} ", i + 1)
            } else {
                String::new()
            };
            println!(
                "info depth {} seldepth {} {}score {} nodes {} nps {} hashfull {} time {} pv {}",
                depth,
                line.sel_depth.max(depth as usize),
                multipv,
                score_str,
                nodes,
                nps,
                hashfull,
                elapsed,
                pv
            );
        }
    }
}

/// Runs one search to completion and prints `bestmove`.
#[allow(clippy::too_many_arguments)]
pub fn go(
    searchers: &mut [Searcher],
    board: &Board,
    limits: Limits,
    tt: &Tt,
    stop: &AtomicBool,
    pondering: &AtomicBool,
    ponder_clock: &Mutex<Instant>,
    overhead: i64,
    multipv: usize,
    ponder_option: bool,
    start: Instant,
) {
    let (soft_ms, hard_ms) = time_limits(&limits, board.stm, overhead);
    tt.new_search();

    let sh = Shared {
        tt,
        stop,
        pondering,
        ponder_clock: limits.ponder.then_some(ponder_clock),
        finished: AtomicBool::new(false),
        nodes: AtomicU64::new(0),
        start,
        soft_ms,
        hard_ms,
        limits,
        best: Mutex::new((Move::NONE, 0, 0)),
    };

    // Only the reporting thread searches more than one line.
    for (i, s) in searchers.iter_mut().enumerate() {
        s.board = board.clone();
        s.multipv = if i == 0 { multipv.max(1) } else { 1 };
    }

    // Lazy SMP: every thread searches the same position on the shared table,
    // and they diverge through the table and through timing. Only thread zero
    // owns the clock, the node limit and reporting; the helpers stop when it
    // raises `stop`.
    let n = searchers.len();
    let (first, rest) = searchers.split_at_mut(1);
    std::thread::scope(|scope| {
        let sh = &sh;
        for s in rest.iter_mut() {
            s.quiet = true;
            // Search with fewer threads rather than bring the engine down.
            if std::thread::Builder::new()
                .stack_size(HELPER_STACK)
                .spawn_scoped(scope, move || {
                    s.iterate(sh);
                })
                .is_err()
            {
                println!("info string could not start a helper thread");
            }
        }
        first[0].iterate(sh);
        sh.finished.store(true, Ordering::Relaxed);
    });

    let mut best = first[0].best_move;
    // Let the threads vote: a move counts for more the deeper the iteration
    // that chose it and the better its score. Helpers often see something the
    // main thread missed, which is most of what lazy SMP is worth.
    //
    // Not under MultiPV: `bestmove` has to be the move on the `multipv 1` line
    // the GUI was just shown, and a helper that searched a different set of
    // lines is not entitled to override it.
    if n > 1 && multipv <= 1 {
        let all = || first.iter().chain(rest.iter());
        let min_score = all()
            .filter(|s| s.completed_depth > 0)
            .map(|s| s.final_score)
            .min()
            .unwrap_or(0);
        let mut votes: Vec<(Move, i64)> = Vec::new();
        for s in all() {
            if s.completed_depth == 0 || s.final_move.is_none() {
                continue;
            }
            let w = (s.final_score - min_score + 14) as i64 * s.completed_depth as i64;
            match votes.iter_mut().find(|(m, _)| *m == s.final_move) {
                Some(e) => e.1 += w,
                None => votes.push((s.final_move, w)),
            }
        }
        if let Some(&(m, _)) = votes.iter().max_by_key(|(_, w)| *w) {
            best = m;
        }
    }
    if best.is_none() {
        best = sh.best.lock().unwrap().0;
    }
    if best.is_none() {
        // Nothing completed; fall back to the first legal move so that a
        // `bestmove` is always produced.
        let mut list = MoveList::new();
        generate::<GEN_ALL>(board, &mut list);
        if list.len > 0 {
            best = list.get(0);
        }
    }
    // An explicit limit or a terminal position may finish while still
    // pondering. Keep the completed vote, but do not publish it yet.
    while pondering.load(Ordering::Acquire) && !stop.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let reply = if ponder_option {
        first
            .iter()
            .chain(rest.iter())
            .filter(|s| s.completed_depth > 0)
            .filter_map(|s| s.lines.first())
            .find(|line| line.mv == best && line.pv.first() == Some(&best))
            .and_then(|line| line.pv.get(1).copied())
            .and_then(|candidate| {
                let mut after = board.clone();
                let mut undo = Undo::default();
                after.make_move(best, &mut undo);
                let mut legal = MoveList::new();
                generate::<GEN_ALL>(&after, &mut legal);
                (0..legal.len)
                    .any(|i| legal.get(i) == candidate)
                    .then_some(candidate)
            })
    } else {
        None
    };
    match reply {
        Some(reply) => println!(
            "bestmove {} ponder {}",
            move_to_string(best),
            move_to_string(reply)
        ),
        None => println!("bestmove {}", move_to_string(best)),
    }
    stop.store(true, Ordering::Relaxed);
}
