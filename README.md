# mbca-10x10

A UCI engine for [Grand Chess](https://en.wikipedia.org/wiki/Grand_Chess).

This engine was derived from the normal 8x8 chess engine in The Marvelous Brass
Chessplaying Automaton, an ambitious chess database/playing app I am developing
for Apple platforms. MBCA is still unreleased, but a
[TestFlight beta](https://testflight.apple.com/join/mqfhpmHY) is available.

## Building

From the project root, run:

```sh
cargo build --release
```

The binary ends up in `target/release/mbca-10x10`.

## The opening book

`book/book.bin` is an opening book, and the engine loads one named
`book.bin` sitting beside the executable automatically. It exists for
**variety** rather than strength: the search is deterministic, and Grand Chess
has no body of opening theory to vary the first moves with. `OwnBook`,
`BookFile`, `BookMaxPly`, `BookVariety` and `BookSeed` control it.

## Playing

Sadly, there are not a lot of Grand Chess compatible chess GUIs out there.
The best supported and generally recommended one seems to be
[XBoard/Winboard](https://www.gnu.org/software/xboard/). I haven't tested
`mbca-10x10` in XBoard or Winboard, but it should work.

`mbca-10x10` is sometimes also available as a bot opponent on [PyChess](https://www.pychess.org/@/MBCA-Grand).

## Strength

With so few Grand Chess engines available, it is difficult to estimate the
strength of `mbca-10x10`. In a 2+1 blitz match against Fairy-Stockfish NNUE,
`mbca-10x10` won by 199-1 (+198, -0, =2). This corresponds to an Elo rating
difference of about 900 points, but with such lopsided results, rating
estimates are wildly inaccurate.

## Acknowledgements

+ The neural network powering `mbca-10x10`'s evaluation was trained using Jamie
  Whiting's excellent [bullet](https://github.com/jw1912/bullet) library.
+ The beautiful Grand Chess variant of chess was invented by Christian Freeling,
  who sadly passed away in 2026. His death was what motivated me to make
  this engine. Grand Chess is a great game -- superior to classical chess in
  most ways, if you ask me -- and deserves to be more popular.
