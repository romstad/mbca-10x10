//! Static exchange evaluation.

use crate::attacks::*;
use crate::board::Board;
use crate::params::*;
use crate::squareset::SquareSet;
use crate::types::*;

/// Exchange value of a piece type, in centipawns.
///
/// The king is 0: it can only ever be the final attacker, and that case is
/// handled explicitly at the bottom of the loop rather than by arithmetic.
#[inline(always)]
pub fn see_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => VAL_PAWN,
        PieceType::Knight => VAL_KNIGHT,
        PieceType::Bishop => VAL_BISHOP,
        PieceType::Rook => VAL_ROOK,
        PieceType::Queen => VAL_QUEEN,
        PieceType::Marshal => VAL_MARSHAL,
        PieceType::Cardinal => VAL_CARDINAL,
        PieceType::King => 0,
    }
}

/// Cheapest attacker first.
const LVA_ORDER: [PieceType; 7] = [
    PieceType::Pawn,     // 100
    PieceType::Knight,   // 300
    PieceType::Bishop,   // 350
    PieceType::Rook,     // 550
    PieceType::Cardinal, // 700  <- discriminant 6, ahead of Marshal's 5
    PieceType::Marshal,  // 850
    PieceType::Queen,    // 1000
];

#[inline]
fn value_of(p: Piece) -> i32 {
    if p.is_none() {
        0
    } else {
        see_value(p.piece_type())
    }
}

/// Diagonal riders of both colours: the pieces a bishop-line reveal can find.
#[inline(always)]
fn diagonal_riders(b: &Board) -> SquareSet {
    b.pieces(PieceType::Bishop) | b.pieces(PieceType::Queen) | b.pieces(PieceType::Cardinal)
}

/// Straight riders of both colours: the pieces a rook-line reveal can find.
#[inline(always)]
fn straight_riders(b: &Board) -> SquareSet {
    b.pieces(PieceType::Rook) | b.pieces(PieceType::Queen) | b.pieces(PieceType::Marshal)
}

pub fn see_ge(b: &Board, m: Move, threshold: i32) -> bool {
    debug_assert!(b.is_pseudo_legal(m), "see_ge on a move that is not legal");
    let (from, to) = (m.from(), m.to());
    let moved = b.piece_at(from);

    let (captured_value, moving_value, mut occ) = if m.is_en_passant() {
        let cap = Board::ep_victim(b.stm, to);
        (
            VAL_PAWN,
            VAL_PAWN,
            b.occupied() - SquareSet::from_square(cap),
        )
    } else if m.is_promotion() {
        // The piece left standing on `to` is the promoted one, so that is what
        // a recapture wins and what the swap-off must value.
        let promo = see_value(m.promo());
        (
            value_of(b.piece_at(to)) + promo - VAL_PAWN,
            promo,
            b.occupied(),
        )
    } else {
        // A move that stays a pawn inside the optional promotion zone lands
        // here, and that is correct: the piece standing on `to` afterwards *is*
        // a pawn. Keying this off the destination rank rather than off
        // `is_promotion()` would value it as a queen and invert the exchange.
        (value_of(b.piece_at(to)), value_of(moved), b.occupied())
    };

    let mut swap = captured_value - threshold;
    if swap < 0 {
        return false;
    }
    swap = moving_value - swap;
    if swap <= 0 {
        return true;
    }

    occ = (occ - SquareSet::from_square(from)) | SquareSet::from_square(to);
    let mut attackers = b.attackers_to(to, occ) & occ;
    let mut stm = b.stm;
    let mut result = true;

    loop {
        stm = stm.flip();
        attackers &= occ;
        let mine = attackers & b.colors(stm);
        if mine.is_empty() {
            break;
        }
        result = !result;

        let mut done = false;
        for pt in LVA_ORDER {
            let set = mine & b.pieces(pt);
            if set.is_empty() {
                continue;
            }
            swap = see_value(pt) - swap;
            if swap < result as i32 {
                return result;
            }
            occ -= SquareSet::from_square(set.lsb());
            // Reveal riders behind the piece that just left. Computed from `to`
            // against the updated occupancy, so it is correct whether the
            // capturer rode or leapt to get here.
            if matches!(
                pt,
                PieceType::Pawn | PieceType::Bishop | PieceType::Cardinal | PieceType::Queen
            ) {
                attackers |= bishop_attacks(to, occ) & diagonal_riders(b);
            }
            if matches!(pt, PieceType::Rook | PieceType::Marshal | PieceType::Queen) {
                attackers |= rook_attacks(to, occ) & straight_riders(b);
            }
            done = true;
            break;
        }
        if !done {
            // Only the king is left: it may capture only if the square is then
            // undefended.
            return if (attackers & occ & b.colors(stm.flip())).any() {
                !result
            } else {
                result
            };
        }
    }
    result
}
