//! The search constants.
//!
//! **Generated. Do not edit by hand.**

// -- piece values -------------------------------------------------------
//
// Used by SEE, by MVV in move ordering, and by the placeholder material
// evaluation, so they are one scale rather than three.
pub const VAL_PAWN: i32 = 100;
pub const VAL_KNIGHT: i32 = 298;
pub const VAL_BISHOP: i32 = 345;
pub const VAL_ROOK: i32 = 542;
pub const VAL_QUEEN: i32 = 999;
pub const VAL_MARSHAL: i32 = 860;
pub const VAL_CARDINAL: i32 = 703;

// -- whole-node pruning -------------------------------------------------
pub const RFP_DEPTH: i32 = 6;
pub const RFP_MARGIN: i32 = 62;
pub const RAZOR_DEPTH: i32 = 3;
pub const RAZOR_BASE: i32 = 308;
pub const RAZOR_SLOPE: i32 = 442;
pub const NMP_MIN_DEPTH: i32 = 3;
pub const NMP_BASE: i32 = 5;
pub const NMP_DEPTH_DIV: i32 = 3;
pub const NMP_EVAL_DIV: i32 = 217;
pub const NMP_EVAL_MAX: i32 = 3;
pub const NMP_VERIFY_SLOPE: i32 = 16;
pub const PROBCUT_DEPTH: i32 = 5;
pub const PROBCUT_RED: i32 = 3;
pub const PROBCUT_MARGIN: i32 = 209;
pub const PROBCUT_IMPROVING: i32 = 52;
pub const IIR_DEPTH: i32 = 2;

// -- move-level pruning -------------------------------------------------
pub const LMP_BASE: i32 = 3;
pub const FP_MAX_DEPTH: i32 = 8;
pub const FP_BASE: i32 = 150;
pub const FP_SLOPE: i32 = 106;
pub const HP_MAX_DEPTH: i32 = 4;
pub const HP_SLOPE: i32 = 3579;
pub const SEE_QUIET: i32 = 31;
pub const SEE_NOISY_DEPTH: i32 = 7;
pub const SEE_NOISY: i32 = 89;
pub const CFP_DEPTH: i32 = 7;
pub const CFP_BASE: i32 = 139;
pub const CFP_SLOPE: i32 = 151;
pub const CFP_HIST_DIV: i32 = 5;
pub const BNFP_DEPTH: i32 = 9;
pub const BNFP_BASE: i32 = 63;
pub const BNFP_SLOPE: i32 = 91;

// -- reductions and extensions ------------------------------------------
pub const LMR_BASE: i32 = 87;
pub const LMR_DIV: i32 = 231;
pub const LMR_D_POW: i32 = 100;
pub const LMR_M_POW: i32 = 100;
pub const LMR_HIST_DIV: i32 = 5956;
pub const LMR_CAPT_DIV: i32 = 5679;
pub const SE_MIN_DEPTH: i32 = 6;
pub const SE_BETA_SLOPE: i32 = 2;
pub const SE_DOUBLE_MARGIN: i32 = 11;
pub const CHECK_EXT_DEPTH: i32 = 5;

// -- history ------------------------------------------------------------
pub const HIST_BONUS_SLOPE: i32 = 137;
pub const HIST_BONUS_OFF: i32 = 118;
pub const HIST_MALUS_SLOPE: i32 = 149;
pub const HIST_MALUS_OFF: i32 = 55;
pub const OPP_HIST_MUL: i32 = 13;
pub const OPP_HIST_MIN: i32 = -1556;
pub const OPP_HIST_MAX: i32 = 1478;
pub const OPP_HIST_OFF: i32 = 596;

// -- quiescence ---------------------------------------------------------
pub const QS_FUTILITY: i32 = 174;
pub const QS_SEE: i32 = 34;
pub const QS_PROMO_DELTA: i32 = 1;
pub const UNDERPROMO_PRUNE_DEPTH: i32 = 0;

// -- evaluation ---------------------------------------------------------
pub const EVAL_FIFTY_SCALE: i32 = 214;
pub const CORR_PAWN_W: i32 = 3;
pub const CORR_CONT_W: i32 = 2;

// -- time management ----------------------------------------------------
pub const TM_SOFT_DIV: i32 = 22;
pub const TM_INC_NUM: i32 = 72;
pub const TM_HARD_DIV: i32 = 3;
pub const TM_HARD_MULT: i32 = 5;
pub const TM_NODE_BASE: i32 = 125;
pub const TM_NODE_SLOPE: i32 = 92;
pub const TM_STAB_SCALE: i32 = 91;
pub const TM_MIN_DEPTH: i32 = 4;
pub const ASP_MIN_DEPTH: i32 = 4;
pub const ASPIRATION_DELTA: i32 = 10;
