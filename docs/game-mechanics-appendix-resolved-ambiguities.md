# Generals.io Mechanics — Appendix: Resolved Ambiguities

**Date:** 2026.05.11
**Status:** Historical reference. Records mechanics details pinned to specific bundle line refs in the live JS bundle (`research/gior-format/generals-main-prod-v31.4.1-d51b92c0.js`, v31.4.1, replay format v18) and empirical replay data. Most entries below resolve a question once flagged as unconfirmed in `generals-io-game-mechanics.md`; the last is an implementation edge case the main doc doesn't elaborate on but that simulator and parser implementors will hit. The resolved rules are folded into the main mechanics doc; this file preserves the trace and provides bundle line refs for developers implementing a replay parser or simulator.

A caveat that motivated keeping this appendix: the JS bundle is the replay-viewer's reconstruction of game logic, not the server's authoritative source. Historical bugs exist. Treat the bundle as the best implementation reference we have, but verify against empirical replay data where it matters.

---

## 1. Convergent moves and ties

**Original question:** when two players move onto the same neutral tile simultaneously and their armies differ by exactly 1, does the larger army capture with 1 remaining, or is it treated as a tie?

**Resolution:** the larger source-army's move resolves first, capturing the neutral tile. The smaller-army move then attacks the now-occupied tile per the normal combat rules. The capturing player holds the tile with armies equal to the difference between sent armies; on exact ties of source army, the first-resolved mover holds the tile with 0 armies (defender's advantage in tied combat).

**Bundle reference:** the priority sort in `MoveResolver.determineMoveOrder` (line 67375–67421) orders the simultaneous moves; `attack()` in the map class (line 67283–67313) performs each combat resolution sequentially.

**Folded into the main doc:** §6 "Move priority order" and "Convergent moves."

---

## 2. Two simultaneous attackers on one defender

**Original question:** when two players simultaneously attack a tile owned by a third player, the exact resolution order is unknown.

**Resolution:** same mechanism as (1) — the priority sort orders the attacking moves and they resolve sequentially. Each successive attacker sees the post-resolution state of the defending tile.

**Bundle reference:** same as (1).

**Folded into the main doc:** §6 "Move priority order" and §7 "Multiple simultaneous attackers."

---

## 3. Production timing on capture

**Original question:** if a city is captured on the same timestep as a turn boundary or land tick, does it produce for the new owner that turn?

**Resolution: yes.** Within a single timestep, the simulator's `update()` resolves all moves first, then increments the turn counter, then runs production for the post-move state. A city or general captured this timestep is owned by the new player when production fires.

**Bundle reference:** `update()` at line 67836–67955 sequences move resolution → `this.turn++` → production loop (generals + cities every 2 timesteps; every owned tile every 50 timesteps).

**Folded into the main doc:** §4 Army generation (the existing description already implies this; the appendix records the implementation reference).

---

## 4. Surrender countdown duration

**Original question:** the countdown was believed to be ~10–15 turns; the exact value was unconfirmed.

**Resolution: 50 timesteps (= 25 turns / 25 seconds at normal game speed; one full round).** From `_goAFK` at line 73439–73489: `neutralizeTurn = afkTurn + floor(2 * game_speed * TIMEOUT_CAPTURE_AFK / 1000)`, where `TIMEOUT_CAPTURE_AFK = 25000` (ms) and `game_speed = 1` for all FFA ladder games (empirically confirmed across a 5000-game sample).

**Two empirical wrinkles** worth noting for the simulator implementor:

- **The replay records a second AFK event when the countdown fires** (via live-class `tryNeutralizePlayer` calling `replay.addAFK` at line 73545). So the countdown is **data-driven from the `afks` array** — no need to hardcode the constant; the replay tells you when each event fires.
- **Only ~37% of AFK'd players see the countdown complete.** The remaining ~63% have only the kill event in the replay — either the game ended before the countdown expired (common in late-game surrenders that drop the game to one remaining player) or another player captured the surrendered player during the countdown. The capture path goes through `executePlayerCapture` at line 68233–68282 and pre-empts neutralization without emitting a second AFK event.

**Folded into the main doc:** §9 "Surrender / disconnect."

---

## 5. Disconnect trigger

**Original question:** when is a player considered disconnected, and how is this represented in replay data?

**Resolution:** disconnects go through `handleLeave` at line 73569, which calls `_ensurePlayerAfkOrDead(n, true)` → `_goAFK(t, ..., true)`. The `transferImmediate=true` arg sets `neutralizeTurn = afkTurn + 1` instead of `afkTurn + 50` — disconnects have a 1-timestep countdown ("effectively immediate" from a player perspective).

In the replay's `afks` array, a disconnect would in principle appear as a paired event with gap=1 (vs. gap=50 for a surrender). **Empirically, zero gap=1 events appear in a 5000-game sample** — disconnects either don't happen in ladder FFA, or the 1-step countdown is always pre-empted by capture or game-end. The disconnect-via-gap-size signal is therefore not observable in practice; treat both paths identically when parsing.

**Folded into the main doc:** §9 "Surrender / disconnect" (covers disconnect path briefly without belaboring the unobservable gap-size distinction).

---

## 6. Chained captures in a single timestep

**Edge case:** when player A captures player B's general *and* player B captures player C's general in the same timestep, what happens?

**Resolution:** the two captures resolve sequentially in the move-priority order described in §6 of the main doc. Both moves are general-attacks, so they share the lowest priority tier and tiebreak on source army size (larger first). The outcome depends on which order they land in.

- **B's source army is larger:** B's move runs first. B captures C and `replaceAll` flips all of C's tiles to B with armies halved. A's move runs next, captures B, and `replaceAll` flips every tile B currently owns — including the tiles B just inherited from C — to A with armies halved again. C's former territory is effectively halved twice (rounded up at each step).
- **A's source army is larger:** A's move runs first. A captures B; `killPlayer(B)` flips all of B's tiles to A. When the loop reaches B's pre-selected move, the per-iteration source-ownership check (`m === this.map.tileAt(f.start)`) fails — B's source tile is now owned by A — and the move is silently skipped. C survives the timestep.

The decisive mechanism is that the move list for the timestep is a snapshot taken once at the top of `update()` via `getNextValidSetOfMovesPrioritized()`, and each move is re-validated against current board state when its turn in the loop comes. `killPlayer` clears the captured player's input buffer for *future* timesteps but does not pull their already-selected move out of the current iteration.

**Bundle reference:** v31.4.1 — `update()` move-list snapshot at line 72060 and per-move ownership check at line 72099; `handleAttack` capture trigger at line 72999–73003; `executePlayerCapture` (`map.replaceAll(e, t, 0.5)`) at line 73194–73195; `killPlayer` (calls `clearMoves`) at line 71955–71966.

**Folded into the main doc:** §9 "Elimination, surrender, and game end" gains a short "Chained captures in one timestep" paragraph linking back here.
