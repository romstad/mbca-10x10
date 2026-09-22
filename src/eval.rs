//! The evaluation the search sees.

use crate::board::{Board, Dirty};
use crate::nnue::NnueState;

/// Incrementally maintained evaluation state, one per searcher. The stack
/// mirrors the search stack: `push` on every move made, `pop` on every move
/// unmade.
pub struct EvalState {
    nnue: NnueState,
}

impl Default for EvalState {
    fn default() -> EvalState {
        EvalState::new()
    }
}

impl EvalState {
    pub fn new() -> EvalState {
        EvalState {
            nnue: NnueState::new(),
        }
    }

    /// Rebuilds from the board. Called at the root of every search.
    pub fn reset(&mut self, b: &Board) {
        self.nnue.reset(b);
    }

    /// Records one move. `b` is the board after it.
    #[inline]
    pub fn push(&mut self, b: &Board, d: &Dirty) {
        self.nnue.push(b, d);
    }

    /// A null move moves no piece, but the stack still grows, because `pop` is
    /// driven by the search and does not know which kind of move it undoes.
    #[inline]
    pub fn push_null(&mut self) {
        self.nnue.push_null();
    }

    #[inline]
    pub fn pop(&mut self) {
        self.nnue.pop();
    }

    /// Side-to-move-relative score, in centipawns.
    #[inline]
    pub fn evaluate(&mut self, b: &Board) -> i32 {
        self.nnue.evaluate(b)
    }
}

/// Full recomputation, for `uci`'s `eval` command and for the `checked`
/// profile's per-node cross-check against the incremental path.
pub fn evaluate_scratch(b: &Board) -> i32 {
    crate::nnue::evaluate_scratch(b)
}
