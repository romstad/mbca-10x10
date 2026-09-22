//! The UCI front end.

use crate::board::Board;
use crate::movegen::{generate, MoveList, GEN_ALL};
use crate::search::{self, Limits, Searcher};
use crate::tt::Tt;
use crate::types::{move_to_string, Move, MAX_PLY};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

pub const NAME: &str = "mbca-10x10";
pub const VERSION: &str = "0.1";
pub const AUTHOR: &str = "Tord";
pub const VARIANT: &str = "grand";

const SEARCH_STACK: usize = 64 * 1024 * 1024;
const MAX_THREADS: usize = 64;
const MAX_MULTIPV: usize = 256;

const DEFAULT_OWN_BOOK: bool = true;
const DEFAULT_BOOK_MAX_PLY: u16 = 16;
const MAX_BOOK_MAX_PLY: u16 = 40;
const DEFAULT_BOOK_VARIETY: i32 = 100;
const MAX_BOOK_VARIETY: i32 = 400;
/// The file the engine looks for when `BookFile` is empty, beside the executable.
const BOOK_BESIDE_EXE: &str = "book.bin";

/// A `type check` value. Anything unrecognised leaves the option alone.
fn parse_check(value: &str, current: bool) -> bool {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => true,
        "false" | "0" | "no" | "off" => false,
        _ => current,
    }
}

/// Parses a move against the legal move list.
pub fn parse_move(b: &Board, s: &str) -> Option<Move> {
    let mut list = MoveList::new();
    generate::<GEN_ALL>(b, &mut list);
    for i in 0..list.len {
        let m = list.get(i);
        if move_to_string(m) == s {
            return Some(m);
        }
    }
    None
}

pub struct Engine {
    board: Board,
    tt: Arc<Tt>,
    pool: Arc<Mutex<Vec<Searcher>>>,
    stop: Arc<AtomicBool>,
    pondering: Arc<AtomicBool>,
    ponder_clock: Arc<Mutex<Instant>>,
    handle: Option<JoinHandle<()>>,
    move_overhead: i64,
    multipv: usize,
    ponder_option: bool,
    book: Option<crate::book::Book>,
    book_source: String,
    own_book: bool,
    book_max_ply: u16,
    book_variety: i32,
    book_rng: crate::rng::Rng,
}

/// Only to satisfy `clippy::new_without_default`. [`Engine::new`] is the
/// constructor.
impl Default for Engine {
    fn default() -> Engine {
        Engine::new()
    }
}

impl Engine {
    /// A book is picked up from beside the executable if there is one.
    pub fn new() -> Engine {
        let mut e = Engine::bare();
        e.find_book_beside_exe();
        e
    }

    fn bare() -> Engine {
        Engine {
            board: Board::startpos(),
            tt: Arc::new(Tt::new(16)),
            pool: Arc::new(Mutex::new(vec![Searcher::new(0)])),
            stop: Arc::new(AtomicBool::new(false)),
            pondering: Arc::new(AtomicBool::new(false)),
            ponder_clock: Arc::new(Mutex::new(Instant::now())),
            handle: None,
            move_overhead: 30,
            multipv: 1,
            ponder_option: false,
            book: None,
            own_book: DEFAULT_OWN_BOOK,
            book_max_ply: DEFAULT_BOOK_MAX_PLY,
            book_variety: DEFAULT_BOOK_VARIETY,
            // Seeded from the clock, so two engines in one match diverge. A
            // fixed `BookSeed` replaces this when a session has to repeat.
            book_rng: crate::rng::Rng::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x5EED),
            ),
            book_source: String::new(),
        }
    }

    /// Looks for [`BOOK_BESIDE_EXE`] next to the running binary.
    fn find_book_beside_exe(&mut self) {
        let Some(path) = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join(BOOK_BESIDE_EXE)))
        else {
            return;
        };
        if !path.is_file() {
            return;
        }
        let name = path.to_string_lossy().to_string();
        if let Ok(b) = crate::book::Book::load(&name) {
            self.book_source = name;
            self.book = Some(b);
        }
    }

    /// Loads the book named by `BookFile`, reporting either way.
    fn load_book(&mut self, path: &str) {
        if path.trim().is_empty() {
            self.book = None;
            self.book_source = String::new();
            self.find_book_beside_exe();
            match &self.book {
                Some(b) => println!(
                    "info string book: {} entries from {}",
                    b.len(),
                    self.book_source
                ),
                None => println!("info string book: none"),
            }
            return;
        }
        match crate::book::Book::load(path) {
            Ok(b) => {
                println!(
                    "info string book: {} entries, {} positions from {path}",
                    b.len(),
                    b.positions()
                );
                self.book = Some(b);
                self.book_source = path.to_string();
            }
            Err(e) => {
                println!("info string book: cannot use {path}: {e}");
                self.book = None;
                self.book_source = String::new();
            }
        }
    }

    /// Waits for the search to finish without asking it to stop.
    fn wait(&mut self) {
        if let Some(h) = self.handle.take() {
            if let Err(panic) = h.join() {
                std::panic::resume_unwind(panic);
            }
        }
    }

    fn stop_and_wait(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.pondering.store(false, Ordering::Relaxed);
        self.wait();
    }

    /// Replays `position [startpos | fen …] [moves …]` onto the board.
    fn set_position(&mut self, tokens: &[&str]) {
        let (mut board, i) = match tokens.first() {
            Some(&"startpos") => (Board::startpos(), 1),
            Some(&"fen") => {
                // The FEN runs until `moves` or the end of the line.
                let end = tokens
                    .iter()
                    .position(|&t| t == "moves")
                    .unwrap_or(tokens.len());
                let fen = tokens[1..end].join(" ");
                match Board::from_fen(&fen) {
                    Ok(b) => (b, end),
                    Err(e) => {
                        println!("info string bad fen: {e}");
                        return;
                    }
                }
            }
            _ => return,
        };
        if tokens.get(i) == Some(&"moves") {
            for t in &tokens[i + 1..] {
                match parse_move(&board, t) {
                    Some(m) => {
                        let mut undo = crate::board::Undo::default();
                        board.make_move(m, &mut undo);
                    }
                    None => {
                        println!("info string illegal move: {t}");
                        return;
                    }
                }
            }
        }
        self.board = board;
    }

    fn parse_go(&self, tokens: &[&str]) -> Limits {
        let mut l = Limits::default();
        let mut i = 0;
        let num = |t: Option<&&str>| -> i64 { t.and_then(|s| s.parse().ok()).unwrap_or(0) };
        while i < tokens.len() {
            match tokens[i] {
                "wtime" => {
                    l.time[0] = num(tokens.get(i + 1));
                    l.has_time = true;
                    i += 1;
                }
                "btime" => {
                    l.time[1] = num(tokens.get(i + 1));
                    l.has_time = true;
                    i += 1;
                }
                "winc" => {
                    l.inc[0] = num(tokens.get(i + 1));
                    i += 1;
                }
                "binc" => {
                    l.inc[1] = num(tokens.get(i + 1));
                    i += 1;
                }
                "movestogo" => {
                    l.movestogo = Some(num(tokens.get(i + 1)));
                    i += 1;
                }
                "movetime" => {
                    l.movetime = Some(num(tokens.get(i + 1)));
                    i += 1;
                }
                // Clamped rather than cast: an unparseable or negative number
                // reads as 0 above, and `depth 0` is a search that returns no
                // move. Unreachable from a sane GUI, but a `go` line built by
                // hand is not bound by that.
                "depth" => {
                    l.depth = Some(num(tokens.get(i + 1)).clamp(1, MAX_PLY as i64) as i32);
                    i += 1;
                }
                "nodes" => {
                    l.nodes = Some(num(tokens.get(i + 1)).max(0) as u64);
                    i += 1;
                }
                "infinite" => l.infinite = true,
                "ponder" => l.ponder = true,
                _ => {}
            }
            i += 1;
        }
        l
    }

    /// Picks a book move for the current position, or `None` to search.
    fn book_move(&mut self, limits: &Limits, ponder: bool) -> Option<crate::book::Choice> {
        if !self.own_book || limits.infinite || ponder {
            return None;
        }
        let book = self.book.as_ref()?;
        if self.board.game_ply > self.book_max_ply {
            return None;
        }
        let probe = book.probe(&self.board);
        for e in &probe.illegal {
            println!(
                "info string book: illegal entry {:#x} for key {:016x}",
                e.mv, e.key
            );
        }
        crate::book::choose(&probe.legal, self.book_variety, &mut self.book_rng)
    }

    fn start_search(&mut self, limits: Limits, start: Instant, ponder: bool) {
        self.stop_and_wait();
        self.stop.store(false, Ordering::SeqCst);
        self.pondering.store(ponder, Ordering::Relaxed);
        *self.ponder_clock.lock().unwrap() = start;

        let pool = Arc::clone(&self.pool);
        let tt = Arc::clone(&self.tt);
        let stop = Arc::clone(&self.stop);
        let pondering = Arc::clone(&self.pondering);
        let ponder_clock = Arc::clone(&self.ponder_clock);
        let board = self.board.clone();
        let overhead = self.move_overhead;
        let multipv = self.multipv;
        let ponder_option = self.ponder_option;
        self.handle = std::thread::Builder::new()
            .stack_size(SEARCH_STACK)
            .spawn(move || {
                let mut guard = pool.lock().unwrap();
                search::go(
                    &mut guard,
                    &board,
                    limits,
                    &tt,
                    &stop,
                    &pondering,
                    &ponder_clock,
                    overhead,
                    multipv,
                    ponder_option,
                    start,
                );
            })
            .ok();
        if self.handle.is_none() {
            println!("info string could not spawn a search thread");
            println!("bestmove 0000");
        }
    }

    fn set_threads(&mut self, n: usize) {
        self.stop_and_wait();
        let threads = n.clamp(1, MAX_THREADS);
        let mut pool = self.pool.lock().unwrap();
        *pool = (0..threads).map(Searcher::new).collect();
    }

    fn set_hash(&mut self, mb: usize) {
        self.stop_and_wait();
        let hash_mb = mb.clamp(1, 65536);
        self.tt = Arc::new(Tt::new(1));
        self.tt = Arc::new(Tt::new(hash_mb));
    }

    fn new_game(&mut self) {
        self.stop_and_wait();
        self.tt.clear();
        let mut pool = self.pool.lock().unwrap();
        for s in pool.iter_mut() {
            s.clear();
        }
    }

    pub fn run(&mut self) {
        let stdin = std::io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            if stdin.read_line(&mut line).unwrap_or(0) == 0 {
                self.stop_and_wait();
                return;
            }
            if !self.execute(&line) {
                return;
            }
        }
    }

    /// Handle one command, exactly as it would have arrived on stdin.
    ///
    /// Returns false for `quit`, which is the stdin loop's signal to stop
    /// reading.
    pub fn execute(&mut self, line: &str) -> bool {
        // Before parsing, so that parse latency counts against the clock.
        let start = Instant::now();
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&cmd) = tokens.first() else {
            return true;
        };
        match cmd {
            "uci" => {
                println!("id name {NAME} {VERSION}");
                println!("id author {AUTHOR}");
                println!("option name UCI_Variant type combo default {VARIANT} var {VARIANT}");
                println!("option name Hash type spin default 16 min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max {MAX_THREADS}");
                println!("option name Move Overhead type spin default 30 min 0 max 5000");
                println!("option name MultiPV type spin default 1 min 1 max {MAX_MULTIPV}");
                println!("option name Ponder type check default false");
                println!("option name Clear Hash type button");
                println!(
                    "option name OwnBook type check default {}",
                    DEFAULT_OWN_BOOK
                );
                println!("option name BookFile type string default");
                println!(
                    "option name BookMaxPly type spin default {DEFAULT_BOOK_MAX_PLY} min 0 max {MAX_BOOK_MAX_PLY}"
                );
                println!(
                    "option name BookVariety type spin default {DEFAULT_BOOK_VARIETY} min 0 max {MAX_BOOK_VARIETY}"
                );
                println!("option name BookSeed type spin default 0 min 0 max 2147483647");
                println!("uciok");
            }
            // Answered without taking the pool lock, so it returns during a
            // search rather than blocking until the search ends.
            "isready" => println!("readyok"),
            "ucinewgame" => {
                self.new_game();
                self.board = Board::startpos();
            }
            "setoption" => self.set_option(&tokens),
            "position" => self.set_position(&tokens[1..]),
            "go" => {
                let limits = self.parse_go(&tokens[1..]);
                let ponder = limits.ponder;
                match self.book_move(&limits, ponder) {
                    Some(c) => {
                        let mv = move_to_string(c.entry.mov());
                        // `depth 0` is the convention for a move that came
                        // from a table rather than a search; a GUI wants
                        // something to display either way.
                        println!(
                            "info depth 0 seldepth 0 score {} nodes 0 nps 0 time 0 pv {mv}",
                            crate::book::score_string(c.entry.score)
                        );
                        println!(
                            "info string book: {} moves, played {mv} weight {}/{}",
                            c.moves, c.weight, c.total
                        );
                        println!("bestmove {mv}");
                    }
                    None => {
                        // Not joined: the stdin loop has `stop` and
                        // `isready` to answer while the search runs, and
                        // the search thread prints `bestmove` itself.
                        self.start_search(limits, start, ponder);
                    }
                }
            }
            "stop" => {
                // Do not join: the search thread prints `bestmove` itself,
                // and blocking here would delay the reply to nothing.
                self.pondering.store(false, Ordering::Relaxed);
                self.stop.store(true, Ordering::SeqCst);
            }
            "ponderhit" => {
                if self.pondering.load(Ordering::Acquire) {
                    *self.ponder_clock.lock().unwrap() = Instant::now();
                    self.pondering.store(false, Ordering::Release);
                }
            }
            "quit" => {
                self.stop_and_wait();
                return false;
            }
            _ => {}
        }
        true
    }

    fn set_option(&mut self, tokens: &[&str]) {
        let name_at = tokens.iter().position(|&t| t == "name");
        let value_at = tokens.iter().position(|&t| t == "value");
        let Some(n) = name_at else { return };
        let end = value_at.unwrap_or(tokens.len());
        if n + 1 > end {
            return;
        }
        let name = tokens[n + 1..end].join(" ").to_ascii_lowercase();
        let value = value_at
            .map(|v| tokens[v + 1..].join(" "))
            .unwrap_or_default();

        match name.as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    self.set_hash(mb);
                }
            }
            "threads" => {
                if let Ok(n) = value.parse::<usize>() {
                    self.set_threads(n);
                }
            }
            "move overhead" => {
                if let Ok(v) = value.parse::<i64>() {
                    self.move_overhead = v.clamp(0, 5000);
                }
            }
            "multipv" => {
                if let Ok(n) = value.parse::<usize>() {
                    self.multipv = n.clamp(1, MAX_MULTIPV);
                }
            }
            "ponder" => self.ponder_option = parse_check(&value, self.ponder_option),
            "clear hash" => self.new_game(),
            "ownbook" => self.own_book = parse_check(&value, DEFAULT_OWN_BOOK),
            "bookfile" => self.load_book(&value),
            "bookmaxply" => {
                if let Ok(n) = value.parse::<u16>() {
                    self.book_max_ply = n.min(MAX_BOOK_MAX_PLY);
                }
            }
            "bookvariety" => {
                if let Ok(v) = value.parse::<i32>() {
                    self.book_variety = v.clamp(0, MAX_BOOK_VARIETY);
                }
            }
            "bookseed" => {
                if let Ok(s) = value.parse::<u64>() {
                    // Zero means "keep drawing from the clock-seeded stream", so
                    // that clearing the option does not silently make a match
                    // deterministic.
                    if s != 0 {
                        self.book_rng = crate::rng::Rng::new(s);
                    }
                }
            }
            "uci_variant" if !value.eq_ignore_ascii_case(VARIANT) => {
                println!("info string unsupported variant: {value}");
            }
            _ => {}
        }
    }
}
