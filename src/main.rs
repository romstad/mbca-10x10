//! Program entry point.

use mbca_grand::{attacks, search, uci};

fn main() {
    // Initialization
    attacks::init();
    search::init();

    // Start UCI
    uci::Engine::new().run()
}
