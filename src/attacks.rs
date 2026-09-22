//! Attack generation: leaper tables, rank lookup, and line attacks for sliders.
//!
//! Everything is built once at start-up into a leaked, immutable table block.

use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Once;

use crate::squareset::SquareSet;
use crate::types::{Color, PieceType, Square};

const DIRS: [(i32, i32); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

const N: usize = 0;
const NE: usize = 1;
const SE: usize = 3;
const S: usize = 4;
const SW: usize = 5;
const NW: usize = 7;

const ROOK_DELTAS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DELTAS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

pub struct Tables {
    knight: [SquareSet; 100],
    king: [SquareSet; 100],
    pawn: [[SquareSet; 100]; 2],
    #[cfg(not(target_arch = "aarch64"))]
    ray: [[SquareSet; 100]; 8],
    rank: [[u16; 1024]; 10],
    #[cfg(target_arch = "aarch64")]
    file_line: [SquareSet; 100],
    #[cfg(target_arch = "aarch64")]
    diag_line: [SquareSet; 100],
    #[cfg(target_arch = "aarch64")]
    anti_line: [SquareSet; 100],
    rook_empty: [SquareSet; 100],
    bishop_empty: [SquareSet; 100],
    between: [[SquareSet; 100]; 100],
    line: [[SquareSet; 100]; 100],
}

static TABLES: AtomicPtr<Tables> = AtomicPtr::new(std::ptr::null_mut());
static ONCE: Once = Once::new();

#[inline(always)]
fn tables() -> &'static Tables {
    unsafe { &*TABLES.load(Ordering::Relaxed) }
}

pub fn init() {
    ONCE.call_once(|| {
        TABLES.store(Box::into_raw(Box::new(Tables::new())), Ordering::Relaxed);
    });
}

fn on_board(f: i32, r: i32) -> bool {
    (0..10).contains(&f) && (0..10).contains(&r)
}

fn sq_at(f: i32, r: i32) -> Square {
    Square::from_file_rank(f as u8, r as u8)
}

/// Squares attacked along `deltas` from `sq`, stopping on (and including) the
/// first occupied square in each direction.
fn sliding_attacks(sq: Square, deltas: &[(i32, i32)], occ: SquareSet) -> SquareSet {
    let mut res = SquareSet::EMPTY;
    for &(df, dr) in deltas {
        let (mut f, mut r) = (sq.file() as i32 + df, sq.rank() as i32 + dr);
        while on_board(f, r) {
            let s = sq_at(f, r);
            res.set(s);
            if occ.has(s) {
                break;
            }
            f += df;
            r += dr;
        }
    }
    res
}

impl Tables {
    fn new() -> Tables {
        let mut knight = [SquareSet::EMPTY; 100];
        let mut king = [SquareSet::EMPTY; 100];
        let mut pawn = [[SquareSet::EMPTY; 100]; 2];
        let mut ray = [[SquareSet::EMPTY; 100]; 8];
        #[cfg(target_arch = "aarch64")]
        let mut file_line = [SquareSet::EMPTY; 100];
        #[cfg(target_arch = "aarch64")]
        let mut diag_line = [SquareSet::EMPTY; 100];
        #[cfg(target_arch = "aarch64")]
        let mut anti_line = [SquareSet::EMPTY; 100];
        let mut rook_empty = [SquareSet::EMPTY; 100];
        let mut bishop_empty = [SquareSet::EMPTY; 100];
        let mut rank = [[0u16; 1024]; 10];
        for (file, entries) in rank.iter_mut().enumerate() {
            for (occupancy, attacks) in entries.iter_mut().enumerate() {
                for step in [-1isize, 1] {
                    let mut f = file as isize + step;
                    while (0..10).contains(&f) {
                        *attacks |= 1 << f;
                        if occupancy & (1 << f) != 0 {
                            break;
                        }
                        f += step;
                    }
                }
            }
        }

        for i in 0..100u8 {
            let s = Square(i);
            let (f, r) = (s.file() as i32, s.rank() as i32);
            for (df, dr) in [
                (1, 2),
                (2, 1),
                (2, -1),
                (1, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
            ] {
                if on_board(f + df, r + dr) {
                    knight[i as usize].set(sq_at(f + df, r + dr));
                }
            }
            for df in -1..=1 {
                for dr in -1..=1 {
                    if (df != 0 || dr != 0) && on_board(f + df, r + dr) {
                        king[i as usize].set(sq_at(f + df, r + dr));
                    }
                }
            }
            let one = SquareSet::from_square(s);
            pawn[0][i as usize] = one.pawn_attacks(Color::White);
            pawn[1][i as usize] = one.pawn_attacks(Color::Black);

            rook_empty[i as usize] = sliding_attacks(s, &ROOK_DELTAS, SquareSet::EMPTY);
            bishop_empty[i as usize] = sliding_attacks(s, &BISHOP_DELTAS, SquareSet::EMPTY);

            for (d, &(df, dr)) in DIRS.iter().enumerate() {
                ray[d][i as usize] = sliding_attacks(s, &[(df, dr)], SquareSet::EMPTY);
            }
            #[cfg(target_arch = "aarch64")]
            {
                let source = SquareSet::from_square(s);
                file_line[i as usize] = source | ray[N][i as usize] | ray[S][i as usize];
                diag_line[i as usize] = source | ray[NE][i as usize] | ray[SW][i as usize];
                anti_line[i as usize] = source | ray[NW][i as usize] | ray[SE][i as usize];
            }
        }

        let mut between = [[SquareSet::EMPTY; 100]; 100];
        let mut line = [[SquareSet::EMPTY; 100]; 100];
        for a in 0..100u8 {
            for b in 0..100u8 {
                if a == b {
                    continue;
                }
                let (sa, sb) = (Square(a), Square(b));
                for deltas in [&ROOK_DELTAS, &BISHOP_DELTAS] {
                    let full = sliding_attacks(sa, deltas, SquareSet::EMPTY);
                    if !full.has(sb) {
                        continue;
                    }
                    between[a as usize][b as usize] =
                        sliding_attacks(sa, deltas, SquareSet::from_square(sb))
                            & sliding_attacks(sb, deltas, SquareSet::from_square(sa));
                    line[a as usize][b as usize] = (full
                        & sliding_attacks(sb, deltas, SquareSet::EMPTY))
                        | SquareSet::from_square(sa)
                        | SquareSet::from_square(sb);
                }
            }
        }

        Tables {
            knight,
            king,
            pawn,
            #[cfg(not(target_arch = "aarch64"))]
            ray,
            rank,
            #[cfg(target_arch = "aarch64")]
            file_line,
            #[cfg(target_arch = "aarch64")]
            diag_line,
            #[cfg(target_arch = "aarch64")]
            anti_line,
            rook_empty,
            bishop_empty,
            between,
            line,
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn ray_pos(d: usize, sq: Square, occ: SquareSet, t: &Tables) -> SquareSet {
    let r = unsafe { *t.ray.get_unchecked(d).get_unchecked(sq.idx()) };
    let blockers = r & occ;
    if blockers.any() {
        r - unsafe { *t.ray.get_unchecked(d).get_unchecked(blockers.lsb().idx()) }
    } else {
        r
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn ray_neg(d: usize, sq: Square, occ: SquareSet, t: &Tables) -> SquareSet {
    let r = unsafe { *t.ray.get_unchecked(d).get_unchecked(sq.idx()) };
    let blockers = r & occ;
    if blockers.any() {
        r - unsafe { *t.ray.get_unchecked(d).get_unchecked(blockers.msb().idx()) }
    } else {
        r
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn hyperbola(sq: Square, occ: SquareSet, line: SquareSet) -> SquareSet {
    let source = 1u128 << sq.0;
    let occupied = (occ.0 & line.0) | source;
    let forward = occupied.wrapping_sub(source << 1);
    let reversed_source = source.reverse_bits();
    let backward = occupied
        .reverse_bits()
        .wrapping_sub(reversed_source << 1)
        .reverse_bits();
    SquareSet((forward ^ backward) & line.0 & !source)
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: SquareSet) -> SquareSet {
    let t = tables();
    let shift = sq.rank() as usize * 10;
    let rank_occ = ((occ.0 >> shift) & 0x3ff) as usize;
    let rank = unsafe {
        *t.rank
            .get_unchecked(sq.file() as usize)
            .get_unchecked(rank_occ)
    };
    let horizontal = SquareSet((rank as u128) << shift);
    #[cfg(target_arch = "aarch64")]
    {
        horizontal | hyperbola(sq, occ, t.file_line[sq.idx()])
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        horizontal | ray_pos(N, sq, occ, t) | ray_neg(S, sq, occ, t)
    }
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: SquareSet) -> SquareSet {
    let t = tables();
    #[cfg(target_arch = "aarch64")]
    {
        hyperbola(sq, occ, t.diag_line[sq.idx()]) | hyperbola(sq, occ, t.anti_line[sq.idx()])
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        ray_pos(NE, sq, occ, t)
            | ray_pos(NW, sq, occ, t)
            | ray_neg(SE, sq, occ, t)
            | ray_neg(SW, sq, occ, t)
    }
}

#[inline(always)]
pub fn rook_rays(sq: Square) -> SquareSet {
    unsafe { *tables().rook_empty.get_unchecked(sq.idx()) }
}

#[inline(always)]
pub fn bishop_rays(sq: Square) -> SquareSet {
    unsafe { *tables().bishop_empty.get_unchecked(sq.idx()) }
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: SquareSet) -> SquareSet {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

#[inline(always)]
pub fn marshal_attacks(sq: Square, occ: SquareSet) -> SquareSet {
    rook_attacks(sq, occ) | knight_attacks(sq)
}

#[inline(always)]
pub fn cardinal_attacks(sq: Square, occ: SquareSet) -> SquareSet {
    bishop_attacks(sq, occ) | knight_attacks(sq)
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> SquareSet {
    unsafe { *tables().knight.get_unchecked(sq.idx()) }
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> SquareSet {
    unsafe { *tables().king.get_unchecked(sq.idx()) }
}

#[inline(always)]
pub fn pawn_attacks(c: Color, sq: Square) -> SquareSet {
    unsafe { *tables().pawn.get_unchecked(c.idx()).get_unchecked(sq.idx()) }
}

/// Attacks of a non-pawn piece type.
#[inline(always)]
pub fn piece_attacks(pt: PieceType, sq: Square, occ: SquareSet) -> SquareSet {
    match pt {
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occ),
        PieceType::Rook => rook_attacks(sq, occ),
        PieceType::Queen => queen_attacks(sq, occ),
        PieceType::Marshal => marshal_attacks(sq, occ),
        PieceType::Cardinal => cardinal_attacks(sq, occ),
        PieceType::King => king_attacks(sq),
        PieceType::Pawn => SquareSet::EMPTY,
    }
}

/// Squares strictly between two aligned squares; empty if they are not aligned.
#[inline(always)]
pub fn between(a: Square, b: Square) -> SquareSet {
    unsafe {
        *tables()
            .between
            .get_unchecked(a.idx())
            .get_unchecked(b.idx())
    }
}

/// The whole line through two aligned squares; empty if they are not aligned.
#[inline(always)]
pub fn line(a: Square, b: Square) -> SquareSet {
    unsafe { *tables().line.get_unchecked(a.idx()).get_unchecked(b.idx()) }
}

#[inline(always)]
pub fn aligned(a: Square, b: Square, c: Square) -> bool {
    line(a, b).has(c)
}
