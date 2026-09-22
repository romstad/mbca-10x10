//! The board: square sets per colour and piece type, a piece array, and the
//! derived state every node needs (checkers, pin blockers, Zobrist keys).

use crate::attacks::*;
use crate::squareset::SquareSet;
use crate::types::*;

// Zobrist keys

pub struct Zobrist {
    /// Indexed by `Piece::idx()`, which runs 0..16 with the `color << 3 | pt`
    /// packing; `Piece::NONE` is 16 and must never reach this table.
    pub psq: [[u64; 100]; 16],
    pub ep: [u64; 100],
    pub side: u64,
}

const fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

static ZOBRIST: Zobrist = {
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut psq = [[0u64; 100]; 16];
    let mut p = 0;
    while p < 16 {
        let mut s = 0;
        while s < 100 {
            psq[p][s] = splitmix(&mut seed);
            s += 1;
        }
        p += 1;
    }
    let mut ep = [0u64; 100];
    let mut s = 0;
    while s < 100 {
        ep[s] = splitmix(&mut seed);
        s += 1;
    }
    Zobrist {
        psq,
        ep,
        side: splitmix(&mut seed),
    }
};

pub const STARTPOS: &str =
    "r8r/1nbqkcabn1/pppppppppp/10/10/10/10/PPPPPPPPPP/1NBQKCABN1/R8R w - - 0 1";

// ---------------------------------------------------------------------------
// Undo information
// ---------------------------------------------------------------------------

/// Up to two pieces removed and one added by a move, for incremental NNUE.
///
/// The array is sized 2 for both, as in orthodox chess, but with castling gone
/// nothing ever adds more than one piece.
#[derive(Clone, Copy, Default)]
pub struct Dirty {
    pub n_sub: u8,
    pub n_add: u8,
    pub sub: [(Piece, Square); 2],
    pub add: [(Piece, Square); 2],
}

#[derive(Clone, Copy)]
pub struct Undo {
    pub ep: Square,
    pub halfmove: u16,
    pub plies_from_null: u16,
    pub key: u64,
    pub pawn_key: u64,
    pub captured: Piece,
    pub capture_sq: Square,
    pub checkers: SquareSet,
    pub blockers: [SquareSet; 2],
    pub check_squares: [SquareSet; 8],
    pub dirty: Dirty,
}

impl Default for Undo {
    fn default() -> Undo {
        Undo {
            ep: Square::NONE,
            halfmove: 0,
            plies_from_null: 0,
            key: 0,
            pawn_key: 0,
            captured: Piece::NONE,
            capture_sq: Square::NONE,
            checkers: SquareSet::EMPTY,
            blockers: [SquareSet::EMPTY; 2],
            check_squares: [SquareSet::EMPTY; 8],
            dirty: Dirty::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Board {
    by_color: [SquareSet; 2],
    by_type: [SquareSet; 8],
    mailbox: [Piece; 100],
    pub ep: Square,
    pub stm: Color,
    pub halfmove: u16,
    pub game_ply: u16,
    pub plies_from_null: u16,
    pub key: u64,
    pub pawn_key: u64,

    pub checkers: SquareSet,
    /// Pieces of *either* colour standing alone between `c`'s king and an enemy
    /// slider. Intersected with our own pieces it gives pins; intersected with
    /// our pieces using `blockers[them]` it gives discovered-check candidates.
    pub blockers: [SquareSet; 2],
    /// Squares from which a piece of each type would check the side *not* to
    /// move (i.e. the king of the opponent of `stm`).
    pub check_squares: [SquareSet; 8],

    /// Position keys of the whole game so far; the last entry is `key`.
    pub hist: Vec<u64>,
}

impl Board {
    pub fn empty() -> Board {
        Board {
            by_color: [SquareSet::EMPTY; 2],
            by_type: [SquareSet::EMPTY; 8],
            mailbox: [Piece::NONE; 100],
            ep: Square::NONE,
            stm: Color::White,
            halfmove: 0,
            game_ply: 0,
            plies_from_null: 0,
            key: 0,
            pawn_key: 0,
            checkers: SquareSet::EMPTY,
            blockers: [SquareSet::EMPTY; 2],
            check_squares: [SquareSet::EMPTY; 8],
            hist: Vec::with_capacity(1024),
        }
    }

    pub fn startpos() -> Board {
        Board::from_fen(STARTPOS).unwrap()
    }

    // -- accessors ----------------------------------------------------------

    #[inline(always)]
    pub fn colors(&self, c: Color) -> SquareSet {
        self.by_color[c.idx()]
    }
    #[inline(always)]
    pub fn pieces(&self, pt: PieceType) -> SquareSet {
        self.by_type[pt.idx()]
    }
    #[inline(always)]
    pub fn colored(&self, c: Color, pt: PieceType) -> SquareSet {
        self.by_color[c.idx()] & self.by_type[pt.idx()]
    }
    #[inline(always)]
    pub fn occupied(&self) -> SquareSet {
        self.by_color[0] | self.by_color[1]
    }
    /// # Bounds
    ///
    /// Unchecked, and a `Move`'s square fields are seven bits wide, so they can
    /// hold 100..=127: values no legal move ever carries but a corrupt or
    /// colliding transposition-table entry can. Every caller that has not
    /// already run the move past `is_pseudo_legal` must bounds-check first;
    /// this assert is what turns the mistake into a `checked`-profile panic
    /// rather than an out-of-bounds read in release.
    #[inline(always)]
    pub fn piece_at(&self, s: Square) -> Piece {
        debug_assert!(s.idx() < 100, "piece_at off the board: {}", s.0);
        unsafe { *self.mailbox.get_unchecked(s.idx()) }
    }
    #[inline(always)]
    pub fn king_sq(&self, c: Color) -> Square {
        (self.by_color[c.idx()] & self.by_type[PieceType::King.idx()]).lsb()
    }
    /// Bishop, queen, and cardinal.
    #[inline(always)]
    pub fn diagonal_sliders(&self, c: Color) -> SquareSet {
        self.by_color[c.idx()]
            & (self.by_type[PieceType::Bishop.idx()]
                | self.by_type[PieceType::Queen.idx()]
                | self.by_type[PieceType::Cardinal.idx()])
    }
    /// Rook, queen and marshal.
    #[inline(always)]
    pub fn straight_sliders(&self, c: Color) -> SquareSet {
        self.by_color[c.idx()]
            & (self.by_type[PieceType::Rook.idx()]
                | self.by_type[PieceType::Queen.idx()]
                | self.by_type[PieceType::Marshal.idx()])
    }
    /// Everything with a knight's leap: knight, marshal and cardinal.
    #[inline(always)]
    pub fn knight_movers(&self) -> SquareSet {
        self.by_type[PieceType::Knight.idx()]
            | self.by_type[PieceType::Marshal.idx()]
            | self.by_type[PieceType::Cardinal.idx()]
    }
    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers.any()
    }
    /// Number of pieces of both colours, kings included.
    #[inline(always)]
    pub fn piece_count(&self) -> u32 {
        self.occupied().count()
    }
    /// True when `c` has no piece other than king and pawns.
    #[inline(always)]
    pub fn only_pawns(&self, c: Color) -> bool {
        (self.colors(c) - self.pieces(PieceType::Pawn) - self.pieces(PieceType::King)).is_empty()
    }

    // -- promotion inventory -------------------------------------------------

    /// May `c` still promote a pawn to `pt`?
    #[inline(always)]
    pub fn promo_available(&self, c: Color, pt: PieceType) -> bool {
        self.colored(c, pt).count() < PROMO_MAX[pt.idx()]
    }

    /// Bit `pt.idx()` set for each type `c` may still promote to.
    #[inline]
    pub fn promo_inventory(&self, c: Color) -> u8 {
        let mut m = 0u8;
        for pt in PROMO_TYPES {
            if self.promo_available(c, pt) {
                m |= 1u8 << pt.idx();
            }
        }
        m
    }

    // -- piece placement ----------------------------------------------------

    #[inline(always)]
    fn hash_piece(&mut self, p: Piece, s: Square) {
        let k = ZOBRIST.psq[p.idx()][s.idx()];
        self.key ^= k;
        if p.piece_type() == PieceType::Pawn {
            self.pawn_key ^= k;
        }
    }

    #[inline(always)]
    fn put(&mut self, p: Piece, s: Square) {
        debug_assert!(self.piece_at(s).is_none());
        self.by_color[p.color().idx()].set(s);
        self.by_type[p.piece_type().idx()].set(s);
        self.mailbox[s.idx()] = p;
        self.hash_piece(p, s);
    }

    #[inline(always)]
    fn remove(&mut self, s: Square) -> Piece {
        let p = self.piece_at(s);
        debug_assert!(!p.is_none());
        self.by_color[p.color().idx()].clear(s);
        self.by_type[p.piece_type().idx()].clear(s);
        self.mailbox[s.idx()] = Piece::NONE;
        self.hash_piece(p, s);
        p
    }

    #[inline(always)]
    fn relocate(&mut self, from: Square, to: Square) {
        let p = self.piece_at(from);
        let mask = SquareSet::from_square(from) | SquareSet::from_square(to);
        self.by_color[p.color().idx()] ^= mask;
        self.by_type[p.piece_type().idx()] ^= mask;
        self.mailbox[from.idx()] = Piece::NONE;
        self.mailbox[to.idx()] = p;
        self.hash_piece(p, from);
        self.hash_piece(p, to);
    }

    #[inline(always)]
    fn set_ep(&mut self, new: Square) {
        if self.ep != new {
            if self.ep != Square::NONE {
                self.key ^= ZOBRIST.ep[self.ep.idx()];
            }
            if new != Square::NONE {
                self.key ^= ZOBRIST.ep[new.idx()];
            }
            self.ep = new;
        }
    }

    /// The square the pawn captured en passant stands on.
    #[inline(always)]
    pub(crate) fn ep_victim(us: Color, to: Square) -> Square {
        Square(if us == Color::White {
            to.0 - 10
        } else {
            to.0 + 10
        })
    }

    // -- attacks ------------------------------------------------------------

    /// All pieces of both colours attacking `s` given occupancy `occ`.
    #[inline]
    pub fn attackers_to(&self, s: Square, occ: SquareSet) -> SquareSet {
        (pawn_attacks(Color::White, s) & self.colored(Color::Black, PieceType::Pawn))
            | (pawn_attacks(Color::Black, s) & self.colored(Color::White, PieceType::Pawn))
            | (knight_attacks(s) & self.knight_movers())
            | (king_attacks(s) & self.pieces(PieceType::King))
            | (rook_attacks(s, occ)
                & (self.pieces(PieceType::Rook)
                    | self.pieces(PieceType::Queen)
                    | self.pieces(PieceType::Marshal)))
            | (bishop_attacks(s, occ)
                & (self.pieces(PieceType::Bishop)
                    | self.pieces(PieceType::Queen)
                    | self.pieces(PieceType::Cardinal)))
    }

    /// True if `c` attacks `s` with occupancy `occ`.
    #[inline]
    pub fn attacked_by(&self, c: Color, s: Square, occ: SquareSet) -> bool {
        let them = self.colors(c);
        if (pawn_attacks(c.flip(), s) & them & self.pieces(PieceType::Pawn)).any() {
            return true;
        }
        if (knight_attacks(s) & them & self.knight_movers()).any() {
            return true;
        }
        if (king_attacks(s) & them & self.pieces(PieceType::King)).any() {
            return true;
        }
        if (rook_attacks(s, occ) & self.straight_sliders(c)).any() {
            return true;
        }
        (bishop_attacks(s, occ) & self.diagonal_sliders(c)).any()
    }

    /// The pieces standing alone between `c`'s king and an enemy slider: they
    /// may only move along the line of the pin.
    fn slider_blockers(&self, c: Color) -> SquareSet {
        let ksq = self.king_sq(c);
        let them = c.flip();
        let snipers = (rook_rays(ksq) & self.straight_sliders(them))
            | (bishop_rays(ksq) & self.diagonal_sliders(them));
        let occ = self.occupied() - snipers;
        let mut blockers = SquareSet::EMPTY;
        for sniper in snipers {
            let b = between(ksq, sniper) & occ;
            if b.any() && !b.more_than_one() {
                blockers |= b;
            }
        }
        blockers
    }

    /// The part every node needs: what is giving check, and which of our own
    /// pieces are pinned.
    pub fn update_check_info(&mut self) {
        let us = self.stm;
        self.blockers[us.idx()] = self.slider_blockers(us);
        let ksq = self.king_sq(us);
        self.checkers = self.attackers_to(ksq, self.occupied()) & self.colors(us.flip());
    }

    /// The extra information `gives_check` needs. Quiescence never asks, and
    /// neither do nodes that return on a hash hit, so this is deferred until a
    /// node actually starts walking its moves.
    pub fn update_check_squares(&mut self) {
        let them = self.stm.flip();
        self.blockers[them.idx()] = self.slider_blockers(them);

        let them_king = self.king_sq(them);
        let occ = self.occupied();
        let rook = rook_attacks(them_king, occ);
        let bishop = bishop_attacks(them_king, occ);
        let knight = knight_attacks(them_king);
        self.check_squares[PieceType::Pawn.idx()] = pawn_attacks(them, them_king);
        self.check_squares[PieceType::Knight.idx()] = knight;
        self.check_squares[PieceType::Bishop.idx()] = bishop;
        self.check_squares[PieceType::Rook.idx()] = rook;
        self.check_squares[PieceType::Queen.idx()] = bishop | rook;
        self.check_squares[PieceType::Marshal.idx()] = rook | knight;
        self.check_squares[PieceType::Cardinal.idx()] = bishop | knight;
        self.check_squares[PieceType::King.idx()] = SquareSet::EMPTY;
    }

    // -- make / unmake ------------------------------------------------------

    pub fn make_move(&mut self, m: Move, undo: &mut Undo) {
        self.make_move_impl::<true>(m, undo);
    }

    /// Search quiescence never reads check-square masks. Its caller must not
    /// inspect them until it next calls `update_check_squares`.
    pub(crate) fn make_move_without_check_squares(&mut self, m: Move, undo: &mut Undo) {
        self.make_move_impl::<false>(m, undo);
    }

    fn make_move_impl<const SAVE_CHECK_SQUARES: bool>(&mut self, m: Move, undo: &mut Undo) {
        undo.ep = self.ep;
        undo.halfmove = self.halfmove;
        undo.plies_from_null = self.plies_from_null;
        undo.key = self.key;
        undo.pawn_key = self.pawn_key;
        undo.checkers = self.checkers;
        undo.blockers = self.blockers;
        if SAVE_CHECK_SQUARES {
            undo.check_squares = self.check_squares;
        }
        undo.captured = Piece::NONE;
        undo.capture_sq = Square::NONE;
        let d = &mut undo.dirty;
        d.n_sub = 0;
        d.n_add = 0;

        let us = self.stm;
        let them = us.flip();
        let (from, to) = (m.from(), m.to());
        let moved = self.piece_at(from);
        debug_assert!(!moved.is_none() && moved.color() == us);

        self.halfmove = self.halfmove.saturating_add(1);
        self.plies_from_null = self.plies_from_null.saturating_add(1);
        self.game_ply += 1;
        // The en passant right expires unless this move creates a new one.
        let mut new_ep = Square::NONE;

        if m.is_en_passant() {
            let cap_sq = Board::ep_victim(us, to);
            let captured = self.remove(cap_sq);
            self.relocate(from, to);
            undo.captured = captured;
            undo.capture_sq = cap_sq;
            d.sub[0] = (moved, from);
            d.sub[1] = (captured, cap_sq);
            d.add[0] = (moved, to);
            d.n_sub = 2;
            d.n_add = 1;
            self.halfmove = 0;
        } else {
            let captured = self.piece_at(to);
            if !captured.is_none() {
                self.remove(to);
                undo.captured = captured;
                undo.capture_sq = to;
                d.sub[d.n_sub as usize] = (captured, to);
                d.n_sub += 1;
                self.halfmove = 0;
            }
            if m.is_promotion() {
                let promoted = Piece::new(us, m.promo());
                self.remove(from);
                self.put(promoted, to);
                d.sub[d.n_sub as usize] = (moved, from);
                d.n_sub += 1;
                d.add[0] = (promoted, to);
                d.n_add = 1;
            } else {
                self.relocate(from, to);
                d.sub[d.n_sub as usize] = (moved, from);
                d.n_sub += 1;
                d.add[0] = (moved, to);
                d.n_add = 1;
            }
            if moved.piece_type() == PieceType::Pawn {
                self.halfmove = 0;
                if from.rank().abs_diff(to.rank()) == 2 {
                    // The two squares differ by exactly 20, so this is exact.
                    let cross = Square((from.0 + to.0) / 2);
                    // X-FEN: record it only when an enemy pawn could actually
                    // capture there, ignoring whether doing so would expose its
                    // own king. This keeps the hash key sharp for repetitions
                    // and makes our FEN output match the oracle's.
                    if (pawn_attacks(us, cross) & self.colored(them, PieceType::Pawn)).any() {
                        new_ep = cross;
                    }
                }
            }
        }

        self.set_ep(new_ep);
        self.stm = them;
        self.key ^= ZOBRIST.side;
        self.update_check_info();
        self.hist.push(self.key);
    }

    pub fn unmake_move(&mut self, m: Move, undo: &Undo) {
        self.unmake_move_impl::<true>(m, undo);
    }

    pub(crate) fn unmake_move_without_check_squares(&mut self, m: Move, undo: &Undo) {
        self.unmake_move_impl::<false>(m, undo);
    }

    fn unmake_move_impl<const SAVE_CHECK_SQUARES: bool>(&mut self, m: Move, undo: &Undo) {
        self.hist.pop();
        self.stm = self.stm.flip();
        let us = self.stm;
        let (from, to) = (m.from(), m.to());

        if m.is_en_passant() {
            self.relocate(to, from);
            self.put(undo.captured, undo.capture_sq);
        } else {
            if m.is_promotion() {
                self.remove(to);
                self.put(Piece::new(us, PieceType::Pawn), from);
            } else {
                self.relocate(to, from);
            }
            if !undo.captured.is_none() {
                self.put(undo.captured, to);
            }
        }

        self.ep = undo.ep;
        self.halfmove = undo.halfmove;
        self.plies_from_null = undo.plies_from_null;
        self.game_ply -= 1;
        self.key = undo.key;
        self.pawn_key = undo.pawn_key;
        self.checkers = undo.checkers;
        self.blockers = undo.blockers;
        if SAVE_CHECK_SQUARES {
            self.check_squares = undo.check_squares;
        }
    }

    pub fn make_null(&mut self, undo: &mut Undo) {
        undo.ep = self.ep;
        undo.halfmove = self.halfmove;
        undo.plies_from_null = self.plies_from_null;
        undo.key = self.key;
        undo.pawn_key = self.pawn_key;
        undo.checkers = self.checkers;
        undo.blockers = self.blockers;
        undo.check_squares = self.check_squares;
        undo.dirty.n_sub = 0;
        undo.dirty.n_add = 0;

        self.set_ep(Square::NONE);
        self.stm = self.stm.flip();
        self.key ^= ZOBRIST.side;
        self.halfmove = self.halfmove.saturating_add(1);
        self.game_ply += 1;
        self.plies_from_null = 0;
        self.update_check_info();
        self.hist.push(self.key);
    }

    pub fn unmake_null(&mut self, undo: &Undo) {
        self.hist.pop();
        self.stm = self.stm.flip();
        self.ep = undo.ep;
        self.halfmove = undo.halfmove;
        self.plies_from_null = undo.plies_from_null;
        self.game_ply -= 1;
        self.key = undo.key;
        self.pawn_key = undo.pawn_key;
        self.checkers = undo.checkers;
        self.blockers = undo.blockers;
        self.check_squares = undo.check_squares;
    }

    // -- draws --------------------------------------------------------------

    /// How many plies back the current position has occurred before, nearest
    /// first, over at most `limit` plies of history.
    fn repeat_distances(&self, limit: usize) -> impl Iterator<Item = usize> + '_ {
        let n = self.hist.len();
        (4..=limit)
            .step_by(2)
            .filter(move |&i| self.hist[n - 1 - i] == self.key)
    }

    /// A repetition of the current position: one repeat inside the search tree
    /// or two in the game history before it.
    pub fn has_repetition(&self, ply: usize) -> bool {
        let max = (self.halfmove as usize)
            .min(self.plies_from_null as usize)
            .min(self.hist.len().saturating_sub(1));
        let mut count = 0;
        for i in self.repeat_distances(max) {
            if i <= ply {
                return true;
            }
            count += 1;
            if count >= 2 {
                return true;
            }
        }
        false
    }

    /// Insufficient material for either side to mate.
    pub fn insufficient_material(&self) -> bool {
        if (self.pieces(PieceType::Pawn)
            | self.pieces(PieceType::Rook)
            | self.pieces(PieceType::Queen)
            | self.pieces(PieceType::Marshal)
            | self.pieces(PieceType::Cardinal))
        .any()
        {
            return false;
        }
        let minors = self.pieces(PieceType::Knight) | self.pieces(PieceType::Bishop);
        minors.count() <= 1
    }

    // -- move validation ----------------------------------------------------

    /// Legality of a pseudo-legal move: does it leave our own king safe?
    ///
    /// The generator already restricts targets so that only three cases reach
    /// here in anger: en passant, king moves, and pinned pieces.
    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.stm;
        let (from, to) = (m.from(), m.to());
        let ksq = self.king_sq(us);

        if m.is_en_passant() {
            let cap_ss = SquareSet::from_square(Board::ep_victim(us, to));
            let occ = (self.occupied() - SquareSet::from_square(from) - cap_ss)
                | SquareSet::from_square(to);
            return (self.attackers_to(ksq, occ) & (self.colors(us.flip()) - cap_ss)).is_empty();
        }
        if from == ksq {
            let occ = self.occupied() - SquareSet::from_square(from);
            return !self.attacked_by(us.flip(), to, occ);
        }
        !self.blockers[us.idx()].has(from) || aligned(ksq, from, to)
    }

    /// Whether a move from the hash table or from a killer slot is a legal move
    /// in this position. Must be strict: bad moves here corrupt the search.
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m.is_none() || m == Move::NULL {
            return false;
        }
        let us = self.stm;
        let (from, to) = (m.from(), m.to());
        if from.0 >= 100 || to.0 >= 100 || from == to {
            return false;
        }
        let moved = self.piece_at(from);
        if moved.is_none() || moved.color() != us {
            return false;
        }
        let occ = self.occupied();

        if m.is_en_passant() {
            if moved.piece_type() != PieceType::Pawn || to != self.ep {
                return false;
            }
            if m.is_promotion() || !pawn_attacks(us, from).has(to) {
                return false;
            }
            return self.is_legal(m);
        }

        if self.colors(us).has(to) {
            return false;
        }

        if moved.piece_type() == PieceType::Pawn {
            // Promotion shape. Relative rank 7 and 8 are the optional zone,
            // where both outcomes are legal; 9 is the last rank, where
            // promotion is mandatory.
            let to_rr = to.relative_rank(us);
            if m.is_promotion() {
                if to_rr < 7 {
                    return false;
                }
                // A move that was legal when it was stored can be illegal now,
                // because the piece it promotes to has since been recaptured.
                if !self.promo_available(us, m.promo()) {
                    return false;
                }
            } else if to_rr == 9 {
                return false;
            }
            let ok = if self.colors(us.flip()).has(to) {
                pawn_attacks(us, from).has(to)
            } else {
                let one = SquareSet::from_square(from).forward(us) - occ;
                let two = (one & SquareSet::relative_rank(us, 3)).forward(us) - occ;
                (one | two).has(to)
            };
            if !ok {
                return false;
            }
        } else {
            if m.is_promotion() {
                return false;
            }
            if !piece_attacks(moved.piece_type(), from, occ).has(to) {
                return false;
            }
        }

        // Must resolve an existing check.
        if self.in_check() && moved.piece_type() != PieceType::King {
            if self.checkers.more_than_one() {
                return false;
            }
            let checker = self.checkers.lsb();
            if !(between(self.king_sq(us), checker) | self.checkers).has(to) {
                return false;
            }
        }
        self.is_legal(m)
    }

    // Does `m` give check to the opponent? `m` must be legal.
    pub fn gives_check(&self, m: Move) -> bool {
        let (from, to) = (m.from(), m.to());
        let us = self.stm;
        let them_king = self.king_sq(us.flip());
        let pt = self.piece_at(from).piece_type();

        // Direct check.
        if m.is_promotion() {
            let occ = (self.occupied() - SquareSet::from_square(from)) | SquareSet::from_square(to);
            if piece_attacks(m.promo(), to, occ).has(them_king) {
                return true;
            }
        } else if self.check_squares[pt.idx()].has(to) {
            return true;
        }

        // Discovered check.
        if self.blockers[us.flip().idx()].has(from) && !aligned(them_king, from, to) {
            return true;
        }

        if m.is_en_passant() {
            let cap_sq = Board::ep_victim(us, to);
            let occ =
                (self.occupied() - SquareSet::from_square(from) - SquareSet::from_square(cap_sq))
                    | SquareSet::from_square(to);
            (rook_attacks(them_king, occ) & self.straight_sliders(us)).any()
                || (bishop_attacks(them_king, occ) & self.diagonal_sliders(us)).any()
        } else {
            false
        }
    }

    // -- FEN ----------------------------------------------------------------

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let mut b = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err("FEN needs at least a board and a side to move".into());
        }

        let bytes = parts[0].as_bytes();
        let mut file = 0u8;
        let mut rank = 9u8;
        let mut i = 0;
        while i < bytes.len() {
            let ch = bytes[i];
            if ch == b'/' {
                if rank == 0 {
                    return Err("too many ranks".into());
                }
                if file != 10 {
                    return Err("rank is not ten squares wide".into());
                }
                rank -= 1;
                file = 0;
                i += 1;
            } else if ch.is_ascii_digit() {
                let mut n = 0u32;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    n = n * 10 + (bytes[i] - b'0') as u32;
                    i += 1;
                }
                if n == 0 || file as u32 + n > 10 {
                    return Err("bad empty-square run".into());
                }
                file += n as u8;
            } else {
                let p = Piece::from_char(ch).ok_or("bad piece letter")?;
                if file > 9 {
                    return Err("rank too long".into());
                }
                b.put(p, Square::from_file_rank(file, rank));
                file += 1;
                i += 1;
            }
        }
        if rank != 0 || file != 10 {
            return Err("board needs ten ranks of ten squares".into());
        }

        b.stm = if parts[1].starts_with('b') {
            Color::Black
        } else {
            Color::White
        };
        if b.stm == Color::Black {
            b.key ^= ZOBRIST.side;
        }

        // Field 3 is castling availability, always "-" in Grand Chess. Consume
        // and ignore whatever is there.

        if parts.len() > 3 && parts[3] != "-" {
            if let Some(ep) = Square::parse(parts[3].as_bytes()) {
                if b.ep_target_is_real(ep) {
                    b.ep = ep;
                    b.key ^= ZOBRIST.ep[ep.idx()];
                }
            }
        }

        b.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        let fullmove: u16 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
        b.game_ply = (fullmove.max(1) - 1) * 2 + if b.stm == Color::Black { 1 } else { 0 };
        b.plies_from_null = b.halfmove;

        if b.colored(Color::White, PieceType::King).count() != 1
            || b.colored(Color::Black, PieceType::King).count() != 1
        {
            return Err("each side needs exactly one king".into());
        }
        // A pawn can never stand on the first or last rank: promotion on the
        // last rank is mandatory and no pawn ever moves backwards. Such a
        // position is unreachable, and the generator's relative-rank-9
        // reasoning stops meaning anything on it.
        if (b.pieces(PieceType::Pawn) & (SquareSet::rank(0) | SquareSet::rank(9))).any() {
            return Err("pawn on the first or last rank".into());
        }
        b.update_check_info();
        b.update_check_squares();
        // The side not to move may not be in check.
        if (b.attackers_to(b.king_sq(b.stm.flip()), b.occupied()) & b.colors(b.stm)).any() {
            return Err("side not to move is in check".into());
        }
        b.hist.push(b.key);
        Ok(b)
    }

    /// Validation of a supplied en passant target.
    fn ep_target_is_real(&self, ep: Square) -> bool {
        let us = self.stm;
        if ep.relative_rank(us) != 6 {
            return false;
        }
        if !self.piece_at(ep).is_none() {
            return false;
        }
        let fwd: i16 = if us == Color::White { 10 } else { -10 };
        let land = Square((ep.0 as i16 - fwd) as u8);
        if self.piece_at(land) != Piece::new(us.flip(), PieceType::Pawn) {
            return false;
        }
        let orig = Square((ep.0 as i16 + fwd) as u8);
        if !self.piece_at(orig).is_none() {
            return false;
        }
        (pawn_attacks(us.flip(), ep) & self.colored(us, PieceType::Pawn)).any()
    }
}
