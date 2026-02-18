we saw GLM5 trying to brute-force problems it probably shouldn't have — like generating a specific 3D asset (the GBA console model) from scratch, when the cost of a human sourcing a quality asset was a fraction of the time and tokens spent.

Setting up explicit pause-and-ask thresholds seems important.

1. Goal + phases. Don’t just state the end goal. Break it into phases with “done” criteria.

Phase 1: CPU core — ARM7TDMI decoder/executor
  Done: all ARM/Thumb instructions implemented, unit tests pass
Phase 2: Memory — map, DMA, hardware registers
  Done: read/write tests pass, DMA functional
Phase 3: Graphics — PPU, modes 0-5
  Done: test ROMs render correctly
2. Conventions. How to work. Be specific — ambiguity here becomes inconsistency over 200 sessions.

- Source in /src, one module per file
- Tests in /tests, mirroring /src
- Run tests after every significant change
- JSDoc on all public functions
3. Notes protocol. This might be the most important part. Context disappears. Notes are how the next session picks up.

After each session, update /notes/progress.md:
1. Completed this session
2. Next steps (specific, actionable)
3. Open questions
4. Blockers
4. Testing gates. Define when to test. Without this, errors compound silently.

One thing we learned: agents tend to gravitate toward tools they’re most familiar with. With a browser skill installed, the model defaulted to using a browser to test JavaScript output. For a system-level build like this, we explicitly directed it to use Node instead. The agent spent a bit more time upfront writing test code to read pixel data via stdio — but the overall process was significantly more streamlined.

- Unit test each instruction after implementation (with node)
- Integration test after each phase (with node)
- Never skip a failing test
- If a fix takes >3 attempts, log and move on
Press enter or click to view image in full size

GLM5 uses node instead of browser to dump framebuffer.
5. Loop breaking. This is what we found separates a 10-minute prompt from a 10-hour one. (Still refining, but the rough idea:)

- Log every retry to /notes/blockers.md with a count
- After 3 failed attempts (check the log), try a different approach
- After 20 min on one issue, log and move on
- If repeating done work, re-read /notes/progress.md
6. Recovery. What to do on a fresh session.

1. Read /notes/progress.md
2. Read /notes/decisions.md
3. Read /notes/blockers.md
4. Check recently modified files
5. Continue from next item

