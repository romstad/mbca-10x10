//! Square sets: a 128-bit set of board squares, one bit per square.
//!
//! Bit `i` corresponds to `Square(i)` in the rank-major a1 = 0 layout, so bits
//! of one rank are contiguous and ascending in file, and the stride between
//! ranks is ten.

use crate::types::{Color, Square};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct SquareSet(pub u128);

const RANK_MASK: u128 = 0x3FF;
const FILE_A_RAW: u128 = {
    let mut m = 0u128;
    let mut r = 0;
    while r < 10 {
        m |= 1u128 << (r * 10);
        r += 1;
    }
    m
};

impl SquareSet {
    pub const EMPTY: SquareSet = SquareSet(0);
    /// Every square of the board: bits 0..100.
    pub const BOARD: SquareSet = SquareSet((1u128 << 100) - 1);
    pub const FILE_A: SquareSet = SquareSet(FILE_A_RAW);
    pub const FILE_J: SquareSet = SquareSet(FILE_A_RAW << 9);

    #[inline(always)]
    pub const fn from_square(s: Square) -> SquareSet {
        debug_assert!(s.0 < 100);
        SquareSet(1u128 << s.0)
    }
    #[inline(always)]
    pub const fn rank(r: u8) -> SquareSet {
        debug_assert!(r < 10);
        SquareSet(RANK_MASK << (r * 10))
    }
    #[inline(always)]
    pub const fn file(f: u8) -> SquareSet {
        debug_assert!(f < 10);
        SquareSet(FILE_A_RAW << f)
    }
    /// Rank `r` counted from `c`'s own side of the board.
    #[inline(always)]
    pub const fn relative_rank(c: Color, r: u8) -> SquareSet {
        match c {
            Color::White => SquareSet::rank(r),
            Color::Black => SquareSet::rank(9 - r),
        }
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
    #[inline(always)]
    pub const fn has(self, s: Square) -> bool {
        debug_assert!(s.0 < 100);
        self.0 & (1u128 << s.0) != 0
    }
    #[inline(always)]
    pub fn set(&mut self, s: Square) {
        debug_assert!(s.0 < 100);
        self.0 |= 1u128 << s.0;
    }
    #[inline(always)]
    pub fn clear(&mut self, s: Square) {
        debug_assert!(s.0 < 100);
        self.0 &= !(1u128 << s.0);
    }
    /// Lowest set square; the set must be non-empty.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(self.0.trailing_zeros() as u8)
    }
    /// Highest set square; the set must be non-empty. Needed by the ray-based
    /// sliding attacks, which scan backwards along the four negative directions.
    #[inline(always)]
    pub const fn msb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square((127 - self.0.leading_zeros()) as u8)
    }
    /// Removes and returns the lowest set square.
    #[inline(always)]
    pub fn pop(&mut self) -> Square {
        let s = self.lsb();
        self.0 &= self.0 - 1;
        s
    }
    #[inline(always)]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    // The eight directional shifts. `BOARD` is applied wherever a square can
    // leave the top of the board; the southward shifts and `east`/`west` cannot
    // produce a bit above 99 from a clean input.
    #[inline(always)]
    pub const fn north(self) -> SquareSet {
        SquareSet((self.0 << 10) & SquareSet::BOARD.0)
    }
    #[inline(always)]
    pub const fn south(self) -> SquareSet {
        SquareSet(self.0 >> 10)
    }
    #[inline(always)]
    pub const fn east(self) -> SquareSet {
        SquareSet((self.0 & !SquareSet::FILE_J.0) << 1)
    }
    #[inline(always)]
    pub const fn west(self) -> SquareSet {
        SquareSet((self.0 & !SquareSet::FILE_A.0) >> 1)
    }
    #[inline(always)]
    pub const fn north_east(self) -> SquareSet {
        SquareSet(((self.0 & !SquareSet::FILE_J.0) << 11) & SquareSet::BOARD.0)
    }
    #[inline(always)]
    pub const fn north_west(self) -> SquareSet {
        SquareSet(((self.0 & !SquareSet::FILE_A.0) << 9) & SquareSet::BOARD.0)
    }
    #[inline(always)]
    pub const fn south_east(self) -> SquareSet {
        SquareSet((self.0 & !SquareSet::FILE_J.0) >> 9)
    }
    #[inline(always)]
    pub const fn south_west(self) -> SquareSet {
        SquareSet((self.0 & !SquareSet::FILE_A.0) >> 11)
    }
    /// One step forward for `c`.
    #[inline(always)]
    pub const fn forward(self, c: Color) -> SquareSet {
        match c {
            Color::White => self.north(),
            Color::Black => self.south(),
        }
    }
    /// The squares attacked by pawns of colour `c` standing on this set.
    #[inline(always)]
    pub const fn pawn_attacks(self, c: Color) -> SquareSet {
        match c {
            Color::White => SquareSet(self.north_east().0 | self.north_west().0),
            Color::Black => SquareSet(self.south_east().0 | self.south_west().0),
        }
    }
}

impl Iterator for SquareSet {
    type Item = Square;
    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pop())
        }
    }
}

macro_rules! binop {
    ($tr:ident, $f:ident, $op:tt) => {
        impl $tr for SquareSet {
            type Output = SquareSet;
            #[inline(always)]
            fn $f(self, rhs: SquareSet) -> SquareSet {
                SquareSet(self.0 $op rhs.0)
            }
        }
    };
}
binop!(BitAnd, bitand, &);
binop!(BitOr, bitor, |);
binop!(BitXor, bitxor, ^);

macro_rules! assignop {
    ($tr:ident, $f:ident, $op:tt) => {
        impl $tr for SquareSet {
            #[inline(always)]
            fn $f(&mut self, rhs: SquareSet) {
                self.0 $op rhs.0;
            }
        }
    };
}
assignop!(BitAndAssign, bitand_assign, &=);
impl std::ops::SubAssign for SquareSet {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: SquareSet) {
        self.0 &= !rhs.0;
    }
}
assignop!(BitOrAssign, bitor_assign, |=);
assignop!(BitXorAssign, bitxor_assign, ^=);

/// Complement *within the board*: the high 28 bits stay clear, so that `!x` can
/// be used as a target mask without a follow-up `& BOARD` at every call site.
impl Not for SquareSet {
    type Output = SquareSet;
    #[inline(always)]
    fn not(self) -> SquareSet {
        SquareSet(!self.0 & SquareSet::BOARD.0)
    }
}

/// Set difference, `a - b`.
impl Sub for SquareSet {
    type Output = SquareSet;
    #[inline(always)]
    fn sub(self, rhs: SquareSet) -> SquareSet {
        SquareSet(self.0 & !rhs.0)
    }
}

impl std::fmt::Display for SquareSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for r in (0..10).rev() {
            for file in 0..10 {
                let s = Square::from_file_rank(file, r);
                f.write_str(if self.has(s) { "X " } else { ". " })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
