//! NNUE evaluation: `(1600 → 1408) × 2 → 8`,
//! SCReLU, trained with bullet and embedded in the executable.

use crate::board::{Board, Dirty};
use crate::types::*;

// -- architecture -----------------------------------------------------------

/// Neurons in the hidden layer, per perspective.
pub const HIDDEN: usize = 1408;

/// Inputs per perspective: 2 (own, enemy) x 8 piece types x 100 squares.
const INPUTS: usize = 1600;

/// Output rows, chosen by piece count when a position is evaluated.
const OUTPUT_BUCKETS: usize = 8;

/// Quantisation: feature weights are scaled by `QA`, output weights by `QB`.
const QA: i32 = 255;
const QB: i32 = 64;

/// Network output to centipawns.
const SCALE: i32 = 400;

const RAW_LEN: usize =
    2 * (INPUTS * HIDDEN + HIDDEN + OUTPUT_BUCKETS * 2 * HIDDEN + OUTPUT_BUCKETS);
const NET_LEN: usize = RAW_LEN.div_ceil(64) * 64;

#[repr(C, align(64))]
struct AlignedBytes([u8; NET_LEN]);

static NET_BYTES: AlignedBytes = AlignedBytes(*include_bytes!("../net/net.bin"));

#[repr(C)]
struct Network {
    feature_weights: [[i16; HIDDEN]; INPUTS],
    feature_bias: [i16; HIDDEN],
    output_weights: [[i16; 2 * HIDDEN]; OUTPUT_BUCKETS],
    output_bias: [i16; OUTPUT_BUCKETS],
}

#[inline(always)]
fn net() -> &'static Network {
    const _: () = assert!(cfg!(target_endian = "little"), "net.bin is little-endian");
    const _: () = assert!(std::mem::size_of::<Network>() == RAW_LEN);
    // SAFETY: the bytes are a flat i16 dump in exactly this layout (see the
    // module doc), `AlignedBytes` gives them at least i16 alignment, and the
    // assertions above pin the size.
    unsafe { &*(NET_BYTES.0.as_ptr() as *const Network) }
}

// -- features ---------------------------------------------------------------

/// A square as `perspective` sees it.
#[inline(always)]
fn orient(perspective: Color, sq: Square) -> usize {
    let s = sq.idx();
    match perspective {
        Color::White => s,
        Color::Black => (9 - s / 10) * 10 + s % 10,
    }
}

/// Whether `perspective`'s features are file-mirrored, its king being on `ksq`.
///
/// The net is trained mirrored: a perspective whose own king stands on files
/// f-j sees the board reflected left to right, so the two halves of the board
/// share one set of weights. A king move across the centre file therefore
/// cannot be applied as a delta: see [`NnueState::push`], which refreshes that
/// perspective from the board instead.
#[inline(always)]
pub fn perspective_mirror(perspective: Color, ksq: Square) -> bool {
    orient(perspective, ksq) % 10 > 4
}

#[inline(always)]
pub fn feature_index(perspective: Color, mirror: bool, p: Piece, sq: Square) -> usize {
    let mut s = orient(perspective, sq);
    if mirror {
        s = s / 10 * 10 + 9 - s % 10;
    }
    let own = if p.color() == perspective { 0 } else { 800 };
    own + 100 * p.piece_type().idx() + s
}

/// The active features of the side to move and of the other side, each sorted.
pub fn active_features(b: &Board) -> [Vec<usize>; 2] {
    let persp = [b.stm, b.stm.flip()];
    let mut out = [Vec::new(), Vec::new()];
    for (i, &c) in persp.iter().enumerate() {
        let mirror = perspective_mirror(c, b.king_sq(c));
        for sq in b.occupied() {
            out[i].push(feature_index(c, mirror, b.piece_at(sq), sq));
        }
        out[i].sort_unstable();
    }
    out
}

// -- accumulators -----------------------------------------------------------

#[derive(Clone)]
#[repr(C, align(64))]
struct Accumulator {
    v: [[i16; HIDDEN]; 2],
    /// Per perspective (indexed by colour): the mirror `v` was built with.
    mirror: [bool; 2],
    /// The move that led here, as feature deltas.
    dirty: Dirty,
    /// Whether `v[colour]` has been brought up to date.
    computed: [bool; 2],
}

impl Accumulator {
    fn new() -> Accumulator {
        Accumulator {
            v: [[0; HIDDEN]; 2],
            mirror: [false; 2],
            dirty: Dirty::default(),
            computed: [false; 2],
        }
    }

    fn refresh_side(&mut self, b: &Board, persp: Color) {
        let n = net();
        let mirror = perspective_mirror(persp, b.king_sq(persp));
        let acc = &mut self.v[persp.idx()];
        acc.copy_from_slice(&n.feature_bias);
        for sq in b.occupied() {
            let w = &n.feature_weights[feature_index(persp, mirror, b.piece_at(sq), sq)];
            for (a, &x) in acc.iter_mut().zip(w.iter()) {
                *a = a.wrapping_add(x);
            }
        }
        self.mirror[persp.idx()] = mirror;
        self.computed[persp.idx()] = true;
    }
}

/// Applies one move's deltas for one perspective.
fn update_side(
    dst: &mut [i16; HIDDEN],
    src: &[i16; HIDDEN],
    d: &Dirty,
    persp: Color,
    mirror: bool,
) {
    let n = net();
    let w = |(p, sq): (Piece, Square)| &n.feature_weights[feature_index(persp, mirror, p, sq)];
    match (d.n_add, d.n_sub) {
        (1, 1) => {
            let (a, s) = (w(d.add[0]), w(d.sub[0]));
            for (((o, &x), &a), &s) in dst.iter_mut().zip(src).zip(a).zip(s) {
                *o = x.wrapping_add(a).wrapping_sub(s);
            }
        }
        (1, 2) => {
            let (a, s0, s1) = (w(d.add[0]), w(d.sub[0]), w(d.sub[1]));
            for ((((o, &x), &a), &s0), &s1) in dst.iter_mut().zip(src).zip(a).zip(s0).zip(s1) {
                *o = x.wrapping_add(a).wrapping_sub(s0).wrapping_sub(s1);
            }
        }
        _ => {
            *dst = *src;
            for &f in &d.add[..d.n_add as usize] {
                for (o, &a) in dst.iter_mut().zip(w(f)) {
                    *o = o.wrapping_add(a);
                }
            }
            for &f in &d.sub[..d.n_sub as usize] {
                for (o, &s) in dst.iter_mut().zip(w(f)) {
                    *o = o.wrapping_sub(s);
                }
            }
        }
    }
}

/// Fuse the final common incremental update with its output dot product.
#[cfg(target_arch = "aarch64")]
fn update_and_flatten<const TWO_SUB: bool>(
    dst: &mut [i16; HIDDEN],
    src: &[i16; HIDDEN],
    d: &Dirty,
    persp: Color,
    mirror: bool,
    output: &[i16],
) -> i32 {
    use std::arch::aarch64::*;
    let n = net();
    let add = &n.feature_weights[feature_index(persp, mirror, d.add[0].0, d.add[0].1)];
    let sub0 = &n.feature_weights[feature_index(persp, mirror, d.sub[0].0, d.sub[0].1)];
    let sub1 = if TWO_SUB {
        &n.feature_weights[feature_index(persp, mirror, d.sub[1].0, d.sub[1].1)]
    } else {
        sub0
    };
    unsafe {
        let zero = vdupq_n_s16(0);
        let qa = vdupq_n_s16(QA as i16);
        let (mut s0, mut s1) = (vdupq_n_s32(0), vdupq_n_s32(0));
        for j in (0..HIDDEN).step_by(8) {
            let mut x = vaddq_s16(
                vld1q_s16(src.as_ptr().add(j)),
                vld1q_s16(add.as_ptr().add(j)),
            );
            x = vsubq_s16(x, vld1q_s16(sub0.as_ptr().add(j)));
            if TWO_SUB {
                x = vsubq_s16(x, vld1q_s16(sub1.as_ptr().add(j)));
            }
            vst1q_s16(dst.as_mut_ptr().add(j), x);
            let v = vminq_s16(vmaxq_s16(x, zero), qa);
            let product = vmulq_s16(v, vld1q_s16(output.as_ptr().add(j)));
            s0 = vmlal_s16(s0, vget_low_s16(v), vget_low_s16(product));
            s1 = vmlal_high_s16(s1, v, product);
        }
        vaddvq_s32(vaddq_s32(s0, s1))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn flatten(acc: &[i16; HIDDEN], w: &[i16]) -> i32 {
    use std::arch::x86_64::*;
    const _: () = assert!(HIDDEN % 16 == 0);
    debug_assert_eq!(w.len(), HIDDEN);
    // SAFETY: `target_feature = "avx2"` is enabled for the whole build, both
    // slices hold HIDDEN i16s, and every load is unaligned.
    unsafe {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut sum = _mm256_setzero_si256();
        let mut j = 0;
        while j < HIDDEN {
            let a = _mm256_loadu_si256(acc.as_ptr().add(j) as *const __m256i);
            let v = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
            let ww = _mm256_loadu_si256(w.as_ptr().add(j) as *const __m256i);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(v, ww), v));
            // A 256-bit register holds sixteen i16s.
            j += 16;
        }
        let lanes: [i32; 8] = std::mem::transmute(sum);
        lanes.iter().fold(0i32, |s, &x| s.wrapping_add(x))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn flatten(acc: &[i16; HIDDEN], w: &[i16]) -> i32 {
    use std::arch::aarch64::*;
    const _: () = assert!(HIDDEN % 16 == 0);
    debug_assert_eq!(w.len(), HIDDEN);
    // SAFETY: both slices hold HIDDEN i16s, and `vld1q` loads are unaligned.
    unsafe {
        let zero = vdupq_n_s16(0);
        let qa = vdupq_n_s16(QA as i16);
        let (mut s0, mut s1, mut s2, mut s3) = (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        );
        let (mut s4, mut s5, mut s6, mut s7) = (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        );
        let mut j = 0;
        while j + 32 <= HIDDEN {
            let a = vld1q_s16(acc.as_ptr().add(j));
            let v = vminq_s16(vmaxq_s16(a, zero), qa);
            let p = vmulq_s16(v, vld1q_s16(w.as_ptr().add(j)));
            s0 = vmlal_s16(s0, vget_low_s16(v), vget_low_s16(p));
            s1 = vmlal_high_s16(s1, v, p);
            let a = vld1q_s16(acc.as_ptr().add(j + 8));
            let v = vminq_s16(vmaxq_s16(a, zero), qa);
            let p = vmulq_s16(v, vld1q_s16(w.as_ptr().add(j + 8)));
            s2 = vmlal_s16(s2, vget_low_s16(v), vget_low_s16(p));
            s3 = vmlal_high_s16(s3, v, p);
            let a = vld1q_s16(acc.as_ptr().add(j + 16));
            let v = vminq_s16(vmaxq_s16(a, zero), qa);
            let p = vmulq_s16(v, vld1q_s16(w.as_ptr().add(j + 16)));
            s4 = vmlal_s16(s4, vget_low_s16(v), vget_low_s16(p));
            s5 = vmlal_high_s16(s5, v, p);
            let a = vld1q_s16(acc.as_ptr().add(j + 24));
            let v = vminq_s16(vmaxq_s16(a, zero), qa);
            let p = vmulq_s16(v, vld1q_s16(w.as_ptr().add(j + 24)));
            s6 = vmlal_s16(s6, vget_low_s16(v), vget_low_s16(p));
            s7 = vmlal_high_s16(s7, v, p);
            j += 32;
        }
        if j < HIDDEN {
            let a = vld1q_s16(acc.as_ptr().add(j));
            let v = vminq_s16(vmaxq_s16(a, zero), qa);
            let p = vmulq_s16(v, vld1q_s16(w.as_ptr().add(j)));
            s0 = vmlal_s16(s0, vget_low_s16(v), vget_low_s16(p));
            s1 = vmlal_high_s16(s1, v, p);
            let a = vld1q_s16(acc.as_ptr().add(j + 8));
            let v = vminq_s16(vmaxq_s16(a, zero), qa);
            let p = vmulq_s16(v, vld1q_s16(w.as_ptr().add(j + 8)));
            s2 = vmlal_s16(s2, vget_low_s16(v), vget_low_s16(p));
            s3 = vmlal_high_s16(s3, v, p);
        }
        let lo = vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3));
        let hi = vaddq_s32(vaddq_s32(s4, s5), vaddq_s32(s6, s7));
        vaddvq_s32(vaddq_s32(lo, hi))
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64"
)))]
#[inline]
fn flatten(acc: &[i16; HIDDEN], w: &[i16]) -> i32 {
    let mut sum = 0i32;
    for (&a, &w) in acc.iter().zip(w.iter()) {
        let v = (a as i32).clamp(0, QA);
        sum += v * v * w as i32;
    }
    sum
}

pub fn output_bucket(b: &Board) -> usize {
    let count = b.piece_count() as usize;
    (count - 2) / 39usize.div_ceil(OUTPUT_BUCKETS)
}

fn evaluate_acc(acc: &Accumulator, b: &Board) -> i32 {
    let n = net();
    let bucket = output_bucket(b);
    let row = &n.output_weights[bucket];
    let (us, them) = (b.stm.idx(), b.stm.flip().idx());
    let sum = flatten(&acc.v[us], &row[..HIDDEN]) + flatten(&acc.v[them], &row[HIDDEN..]);
    (sum / QA + n.output_bias[bucket] as i32) * SCALE / (QA * QB)
}

/// A full recomputation: the `eval` command, and the `checked` profile's
/// reference for the incremental path.
pub fn evaluate_scratch(b: &Board) -> i32 {
    let mut acc = Accumulator::new();
    acc.refresh_side(b, Color::White);
    acc.refresh_side(b, Color::Black);
    evaluate_acc(&acc, b)
}

/// One accumulator per search ply.
pub struct NnueState {
    stack: Vec<Accumulator>,
    top: usize,
}

impl Default for NnueState {
    fn default() -> NnueState {
        NnueState::new()
    }
}

impl NnueState {
    pub fn new() -> NnueState {
        NnueState {
            stack: vec![Accumulator::new(); MAX_PLY + 16],
            top: 0,
        }
    }

    pub fn reset(&mut self, b: &Board) {
        self.top = 0;
        self.stack[0].refresh_side(b, Color::White);
        self.stack[0].refresh_side(b, Color::Black);
    }

    /// Records the position after a move. `b` is the board *after* it.
    pub fn push(&mut self, b: &Board, d: &Dirty) {
        let prev = self.top;
        self.top += 1;
        debug_assert!(self.top < self.stack.len(), "NNUE stack overflow");
        let (head, tail) = self.stack.split_at_mut(self.top);
        let (old, next) = (&head[prev], &mut tail[0]);
        next.dirty = *d;
        next.computed = [false; 2];
        for c in [Color::White, Color::Black] {
            next.mirror[c.idx()] = perspective_mirror(c, b.king_sq(c));
        }
        if next.mirror != old.mirror {
            for c in [Color::White, Color::Black] {
                if next.mirror[c.idx()] != old.mirror[c.idx()] {
                    next.refresh_side(b, c);
                }
            }
        }
    }

    pub fn push_null(&mut self) {
        let prev = self.top;
        self.top += 1;
        let (head, tail) = self.stack.split_at_mut(self.top);
        let next = &mut tail[0];
        next.dirty = Dirty::default();
        next.mirror = head[prev].mirror;
        next.computed = [false; 2];
    }

    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.top > 0, "NNUE stack underflow");
        self.top -= 1;
    }

    /// Replays pending deltas for one perspective from the nearest computed
    /// accumulator below the top.
    fn ensure(&mut self, persp: Color) {
        let p = persp.idx();
        let mut i = self.top;
        while !self.stack[i].computed[p] {
            debug_assert!(i > 0, "no computed accumulator below the top");
            i -= 1;
        }
        while i < self.top {
            let (head, tail) = self.stack.split_at_mut(i + 1);
            let dst = &mut tail[0];
            let (d, mirror) = (dst.dirty, dst.mirror[p]);
            update_side(&mut dst.v[p], &head[i].v[p], &d, persp, mirror);
            dst.computed[p] = true;
            i += 1;
        }
    }

    fn ensure_flatten(&mut self, persp: Color, weights: &[i16]) -> i32 {
        let p = persp.idx();
        #[cfg(target_arch = "aarch64")]
        if self.top > 0 && !self.stack[self.top].computed[p] && self.stack[self.top - 1].computed[p]
        {
            let top = self.top;
            let (head, tail) = self.stack.split_at_mut(top);
            let dst = &mut tail[0];
            let d = dst.dirty;
            let mirror = dst.mirror[p];
            let src = &head[top - 1].v[p];
            let sum = match (d.n_add, d.n_sub) {
                (1, 1) => Some(update_and_flatten::<false>(
                    &mut dst.v[p],
                    src,
                    &d,
                    persp,
                    mirror,
                    weights,
                )),
                (1, 2) => Some(update_and_flatten::<true>(
                    &mut dst.v[p],
                    src,
                    &d,
                    persp,
                    mirror,
                    weights,
                )),
                _ => None,
            };
            if let Some(sum) = sum {
                dst.computed[p] = true;
                return sum;
            }
        }
        self.ensure(persp);
        flatten(&self.stack[self.top].v[p], weights)
    }

    pub fn evaluate(&mut self, b: &Board) -> i32 {
        let n = net();
        let bucket = output_bucket(b);
        let row = &n.output_weights[bucket];
        let sum = self.ensure_flatten(b.stm, &row[..HIDDEN])
            + self.ensure_flatten(b.stm.flip(), &row[HIDDEN..]);
        (sum / QA + n.output_bias[bucket] as i32) * SCALE / (QA * QB)
    }
}
