https://database.lichess.org/#standard_games

Notes
About 6% of the games include Stockfish analysis evaluations: [%eval 2.35] (235 centipawn advantage), [%eval #-4] (getting mated in 4), always from White's point of view.

The WhiteElo and BlackElo tags contain Glicko2 ratings.

Games contain clock information down to the second as PGN %clk comments since April 2017. If you need centisecond precision, there is a separate export of games across all chess variants and time controls from 2013 to 2021 using %clkc comments.

Players using the Bot API are marked with [WhiteTitle "BOT"] or [BlackTitle "BOT"], respectively.

Variant games have a Variant tag, e.g., [Variant "Antichess"].

Decompress .zst
Unix: pzstd -d filename.pgn.zst (faster than unzstd)
Windows: use PeaZip

Expect uncompressed files to be about 7.1 times larger.

ZStandard archives are partially decompressable, so you can start downloading and then cancel at any point. You will be able to decompress the partial download if you only want a smaller set of game data.

You can also decompress the data on-the-fly without having to create large temporary files. This example shows how you can pipe the contents to a Python script for analyzing using zstdcat.

$ zstdcat lichess_db.pgn.zst | python script.py
Open PGN files
Traditional PGN databases, like SCID or ChessBase, fail to open large PGN files. Until they fix it, you can split the PGN files, or use programmatic APIs such as python-chess or Scoutfish.

Known issues
November 2023: Some Chess960 rematches were recorded with invalid castling rights in their starting FEN.
December 2022: Some Antichess games were recorded with bullet ratings.
12th March 2021: Some games have incorrect results due to a database outage in the aftermath of a datacenter fire.
9th February 2021: Some games were resigned even after the game ended. In variants, additional moves could be played after the end of the game.
December 2020, January 2021: Many variant games have been mistakenly analyzed using standard NNUE, leading to incorrect evaluations.
Up to December 2020: Some exports are missing the redundant (but strictly speaking mandatory) Round tag (always -), Date tag (see UTCDate & UTCTime instead), and black move numbers after comments. This may be fixed by a future re-export.
July 2020 (especially 31st), August 2020 (up to 16th): Many games, especially variant games, may have incorrect evaluations in the opening (up to 15 plies).
December 2016 (up to and especially 9th): Many games may have incorrect evaluations.
Before 2016: In some cases, mate may not be forced in the number of moves given by the evaluations.
June 2020, all before March 2016: Some players were able to play themselves in rated games.
Up to August 2016: 7 games with illegal castling moves were recorded.
