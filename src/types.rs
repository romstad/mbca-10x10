//! Basic newtypes: colours, piece types, pieces, squares and moves.

use std::fmt;

/// Hard ceiling on search depth / ply indices.
pub const MAX_PLY: usize = 246;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }
    #[inline(always)]
    pub const fn from_idx(i: usize) -> Color {
        if i == 0 {
            Color::White
        } else {
            Color::Black
        }
    }
}

/// The eight Grand Chess piece types.
///
/// Marshal is rook+knight and Cardinal is bishop+knight. Diagonal sliders are
/// bishop, queen and cardinal; straight sliders are rook, queen and marshal.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    Marshal = 5,
    Cardinal = 6,
    King = 7,
}

pub const PIECE_TYPES: [PieceType; 8] = [
    PieceType::Pawn,
    PieceType::Knight,
    PieceType::Bishop,
    PieceType::Rook,
    PieceType::Queen,
    PieceType::Marshal,
    PieceType::Cardinal,
    PieceType::King,
];

/// The six types a pawn may become, in roughly descending value order.
pub const PROMO_TYPES: [PieceType; 6] = [
    PieceType::Queen,
    PieceType::Marshal,
    PieceType::Cardinal,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
];

/// How many of each type a side may have on the board at once.
///
/// A pawn may promote only to a type its side is currently missing relative to
/// these maxima. Pawn and King sit at 0 so that `promo_available` rejects them
/// without a special case -- which also rejects a garbage promotion field read
/// back out of the transposition table.
pub const PROMO_MAX: [u32; 8] = [0, 2, 2, 2, 1, 1, 1, 0];

impl PieceType {
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }
    #[inline(always)]
    pub const fn from_idx(i: usize) -> PieceType {
        PIECE_TYPES[i]
    }

    /// The serialized letter, lowercase.
    ///
    /// Written out rather than indexed into a byte string because two of these
    /// cross the traditional names: Fairy-Stockfish calls the Marshal a
    /// *chancellor* and the Cardinal an *archbishop*, and this engine adopts its
    /// letters. So `Marshal` prints `c` and `Cardinal` prints `a`, and the
    /// traditional `m` appears nowhere in any serialized form.
    pub const fn to_char(self) -> u8 {
        match self {
            PieceType::Pawn => b'p',
            PieceType::Knight => b'n',
            PieceType::Bishop => b'b',
            PieceType::Rook => b'r',
            PieceType::Queen => b'q',
            PieceType::Marshal => b'c',
            PieceType::Cardinal => b'a',
            PieceType::King => b'k',
        }
    }

    /// The inverse of `to_char`, case-insensitive.
    pub const fn from_char(c: u8) -> Option<PieceType> {
        Some(match c.to_ascii_lowercase() {
            b'p' => PieceType::Pawn,
            b'n' => PieceType::Knight,
            b'b' => PieceType::Bishop,
            b'r' => PieceType::Rook,
            b'q' => PieceType::Queen,
            b'c' => PieceType::Marshal,
            b'a' => PieceType::Cardinal,
            b'k' => PieceType::King,
            _ => return None,
        })
    }
}

/// A coloured piece, encoded as `color << 3 | piece_type`; `Piece::NONE` is 16.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece(pub u8);

impl Default for Piece {
    fn default() -> Piece {
        Piece::NONE
    }
}

impl Piece {
    pub const NONE: Piece = Piece(16);

    #[inline(always)]
    pub const fn new(c: Color, pt: PieceType) -> Piece {
        Piece((c as u8) << 3 | pt as u8)
    }
    #[inline(always)]
    pub const fn color(self) -> Color {
        debug_assert!(self.0 < 16);
        Color::from_idx((self.0 >> 3) as usize)
    }
    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        debug_assert!(self.0 < 16);
        PieceType::from_idx((self.0 & 7) as usize)
    }
    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == Piece::NONE.0
    }
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    pub fn to_char(self) -> u8 {
        if self.is_none() {
            return b'.';
        }
        let c = self.piece_type().to_char();
        if self.color() == Color::White {
            c.to_ascii_uppercase()
        } else {
            c
        }
    }

    pub fn from_char(c: u8) -> Option<Piece> {
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        Some(Piece::new(color, PieceType::from_char(c)?))
    }
}

/// A board square in rank-major, a1 = 0 numbering.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash, Default)]
pub struct Square(pub u8);

impl Square {
    pub const NONE: Square = Square(100);

    /// `file` and `rank` are both 0-based; rank 0 is rank 1.
    #[inline(always)]
    pub const fn from_file_rank(file: u8, rank: u8) -> Square {
        debug_assert!(file < 10 && rank < 10);
        Square(rank * 10 + file)
    }
    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 % 10
    }
    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 / 10
    }
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
    /// Rank as seen by `c` (0 = that side's first rank).
    #[inline(always)]
    pub const fn relative_rank(self, c: Color) -> u8 {
        match c {
            Color::White => self.rank(),
            Color::Black => 9 - self.rank(),
        }
    }

    /// Parses a square at the start of `s`, returning it and the bytes consumed.
    ///
    /// Ranks may be two digits, so squares are not fixed width and the caller
    /// cannot slice at a known offset.
    pub fn parse_prefix(s: &[u8]) -> Option<(Square, usize)> {
        if s.len() < 2 {
            return None;
        }
        let file = s[0].wrapping_sub(b'a');
        if file >= 10 {
            return None;
        }
        if s[1] == b'1' && s.get(2) == Some(&b'0') {
            Some((Square::from_file_rank(file, 9), 3))
        } else {
            let rank = s[1].wrapping_sub(b'1');
            if rank < 9 {
                Some((Square::from_file_rank(file, rank), 2))
            } else {
                None
            }
        }
    }

    /// Parses a square that must consume all of `s`.
    pub fn parse(s: &[u8]) -> Option<Square> {
        match Square::parse_prefix(s) {
            Some((sq, n)) if n == s.len() => Some(sq),
            _ => None,
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Square::NONE {
            return write!(f, "-");
        }
        // `{}` on the u8 prints rank 10 as two characters; `b'1' + rank` would
        // print ':'.
        write!(f, "{}{}", (b'a' + self.file()) as char, self.rank() + 1)
    }
}

/// Move encoding: `from | to << 7 | promo << 14 | ep_flag << 17`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move(pub u32);

const EP_FLAG: u32 = 1 << 17;

impl Move {
    pub const NONE: Move = Move(0);
    /// A distinct non-move used as a null-move marker in the search stack.
    pub const NULL: Move = Move(u32::MAX);

    #[inline(always)]
    pub const fn new(from: Square, to: Square) -> Move {
        Move(from.0 as u32 | (to.0 as u32) << 7)
    }
    #[inline(always)]
    pub const fn promotion(from: Square, to: Square, promo: PieceType) -> Move {
        debug_assert!(promo as u8 != 0 && (promo as u8) < 7);
        Move(from.0 as u32 | (to.0 as u32) << 7 | (promo as u32) << 14)
    }
    #[inline(always)]
    pub const fn en_passant(from: Square, to: Square) -> Move {
        Move(from.0 as u32 | (to.0 as u32) << 7 | EP_FLAG)
    }
    #[inline(always)]
    pub const fn from(self) -> Square {
        Square((self.0 & 0x7F) as u8)
    }
    #[inline(always)]
    pub const fn to(self) -> Square {
        Square(((self.0 >> 7) & 0x7F) as u8)
    }
    #[inline(always)]
    pub const fn promo_bits(self) -> u8 {
        ((self.0 >> 14) & 7) as u8
    }
    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        self.promo_bits() != 0
    }
    #[inline(always)]
    pub const fn promo(self) -> PieceType {
        PieceType::from_idx(self.promo_bits() as usize)
    }
    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        self.0 & EP_FLAG != 0
    }
    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// Long algebraic notation, e.g. `e7e8`, `e7e8c`, `e9e10a`.
pub fn move_to_string(m: Move) -> String {
    if m.is_none() {
        return "0000".to_string();
    }
    let mut s = format!("{}{}", m.from(), m.to());
    if m.is_promotion() {
        s.push(m.promo().to_char() as char);
    }
    s
}
