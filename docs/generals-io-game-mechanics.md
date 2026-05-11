# Generals.io: Game Mechanics Reference

**Date:** 2026.05.03

**Purpose:** Canonical reference for the core mechanics of generals.io, written to support neural network architecture and replay parser design. Other project docs assume familiarity with this material.

**Scope:** This document covers the standard FFA and 1v1 game mode. It does not cover team modes, custom games, special events, or other variants.

**Appendix:** [`game-mechanics-appendix-resolved-ambiguities.md`](./game-mechanics-appendix-resolved-ambiguities.md) — implementation-reference notes on mechanics details that were once unconfirmed and have since been resolved against the live JS client bundle.

---

## 1. Overview

Generals.io is a real-time strategy game played on a grid.

* Each player starts with a single tile — their general — and expands outward by moving armies into adjacent tiles.
* Goal: find and capture the enemy generals.
* A player is eliminated when their general is captured.
* The game ends when only one general remains.

The core loop:

* Explore the map to discover opponents
* Expand territory to generate armies
* Attack opponents to increase your land, decrease theirs, and to capture their generals

Generals.io is **not** a complete information game

* You can only see what other players are doing in the tiles immediately neighboring your own.
* Everything else is hidden in the fog-of-war.

---

## 2. Map

The map is a rectangular grid of variable dimensions. Each cell is one of the following:

- **Empty tile.** Passable. Starts neutral and unowned.
- **Mountain.** Impassable. Cannot be moved onto or through. Density is approximately 20% of the map but varies by game settings. Mountains shape the effective topology of the board — two tiles that are close on the grid may be far apart in actual traversal distance.
- **City.** A passable structure that produces armies (see §4). Cities come in two forms:
  - **Neutral city.** Unowned at game start, garrisoned with 40–50 neutral armies. Capturing a city costs a large upfront army investment but grants permanent army production. Captured generals also become cities (see §9).
  - **Player-owned city.** A city that has been captured by a player. Produces +1 army per turn for the owning player.
- **General.** Each player's home tile. Functions as a city for army production purposes. Losing your general means elimination.
- **Neutral armies.** A less common tile state created by the surrender mechanic (see §9). When a player surrenders, their non-structure tiles become neutral land that retains its army count. These tiles cost armies to capture or traverse — they behave like low-garrison neutral cities in terms of combat resolution. Neutral armies do not produce additional armies.

**General placement:** the map generator always places generals at least 10 tiles apart (Manhattan distance). This constraint is strategically significant (see §10).

---

## 3. Time structure

The game clock has three levels of time granularity:

- **Timestep.** The atomic unit. One timestep occurs every 500ms (twice per second). Each player may make exactly one move per timestep, or pass.
- **Turn.** 2 timesteps = 1 second. Generals and cities produce armies once per turn (see §4).
- **Round.** 25 turns = 50 timesteps = 25 seconds. The "land tick" occurs once per round (see §4).

When other docs refer to "turn 50" or "1800-turn game," the unit is turns (i.e., pairs of timesteps).

**A naming note: "half-turn" = timestep.** The atomic unit is sometimes called a "half-turn" in the community, and in the project's wire-format and DB schemas the field named `turn` is actually a half-turn (= timestep), not a full turn. Same word, two referents — watch for it when reading data fields vs. mechanics prose.

---

## 4. Army generation

Two distinct production systems run on different timers:

**Per-turn production (every 2 timesteps / 1 second):**
Each general and each owned city produces +1 army on its own tile. A player with 1 general and 3 cities gains +4 army per turn, concentrated on those 4 tiles. This is the steady income stream that makes city count the primary economic variable.

**Per-round production — the "land tick" (every 25 turns / 50 timesteps / 25 seconds):**
Every owned tile gains +1 army. A player with 100 land gains +100 army in a single burst, distributed across all 100 tiles. This is why land count matters strategically — more land means a bigger periodic army spike. The land tick is also the reason expansion is valuable beyond just territorial control.

**Combined effect during peacetime:** between land ticks, a player's army grows at a rate of (1 + city_count) per turn.

---

## 5. Fog of war

**Visibility rule:** a player has Moore-neighborhood visibility (the 8 surrounding cells) around every tile they own. You see the true and current state of any cell within this visibility range — ownership, army count, structure type.

**Everything else is fogged.** Behind fog, enemy positions, army counts, and ownership are unknown.

**Structures have special rules:**

- **Structure existence is always visible** — the existence of a mountain or city is visible regardless of fog status.
- **Behind the fog, mountains and cities are indistinguishable** — when in the fog, both of these structures are displayed with the same icon, making them indistinguishable.
  - You need vision of that tile to know whether it is a mountain (permanently impassable) or city (traversable with a cost).
  - **Structure type never changes** — mountains stay mountains, cities stay cities. Structure type information learned with previous exploration stays accurate.
- **Enemy generals are completely hidden** — unlike mountains and cities, enemy generals behind fog are not visible at all. They appear as ordinary empty tiles. A general is only revealed when a player gains direct vision of it (i.e., owns an adjacent tile). This makes general-hunting a core strategic skill — you must actively explore to find opponents.

**The scoreboard is always visible.** See §8.

**What fog hides:** the positions and army counts of all enemy tiles you don't currently have vision of. A tile you saw 50 turns ago could now have a completely different owner and army count — you have no way to know without regaining vision.

---

## 6. Moves

**Basic structure:** a move consists of a source tile and a cardinal direction (up, down, left, right). The source tile must be owned by the player and must have at least 2 armies (since 1 is always left behind). The player may also pass (make no move). Each player may make at most one move per timestep.

**Normal move:** sends `a - 1` armies from the source tile in the chosen direction, where `a` is the army count on the source tile. Exactly 1 army is always left behind on the source tile to maintain ownership.

**Split move:** each move has a binary split flag. A split move sends `floor(a / 2)` armies instead, leaving `ceil(a / 2)` on the source tile. Split moves allow a player to advance while retaining a meaningful defensive presence, or to branch armies in two directions across consecutive timesteps.

**Moving onto a friendly tile:** armies combine additively. If source has `a` armies and destination has `b` armies (both owned by the same player), the result is 1 army on the source tile, `(a - 1) + b` armies on the destination tile (for a normal move).

**Moving onto a neutral empty tile:** the tile is captured and the arriving armies occupy it. Empty tiles have 0 defending armies, so this always succeeds — it is the standard way players expand territory in the early game. Less commonly, neutral tiles can have nonzero garrisons as a result of a player surrendering (see §9 and "neutral armies" in §2). Moving onto these tiles is resolved as normal combat — the arriving army fights the garrison.

**The "leave 1 behind" rule and its implications:** because movement always leaves at least 1 army on the source tile, non-combat movement never decreases owned territory. Once you own a tile, you own it until an opponent captures it by force. There is no mechanic for voluntarily abandoning territory.

**Simultaneous execution:** all players' moves for a given timestep are submitted simultaneously. When moves interact, they are resolved in a deterministic priority order.

**Move priority order.** Within a single timestep, the simultaneous moves are sorted by:

1. **Defensive moves first.** Moves that move armies onto a tile already owned by the moving player (or teammate) are processed before moves that attack another player.
2. **General-captures last.** Among attacking moves, attacks on an opponent's general tile resolve after other attacking moves.
3. **Larger source army first.** Among moves of the same class, the move originating from the tile with more armies resolves first.
4. **Input order tiebreak.** If still tied, the move submitted earlier in the timestep takes priority.

**The "inward-first" rule:** for any pair of moves m1 and m2, if m1's destination is m2's source tile, then m1 executes before m2 — regardless of where the priority order above would otherwise have placed them. Intuitively, a tile must face an incoming move before it can execute its own outgoing move. This has two important consequences:

- **Defending/chasing is favored.** If m1 attacks the tile that m2 is trying to move away from, m1 lands first. m2's source tile absorbs the hit — its army is reduced (or the tile is captured entirely) — before m2's outgoing move resolves. If m1 captures the tile, m2's move is canceled.
- **The rule chains.** If m3's source tile is m2's destination, then m2 executes before m3 by the same logic. Combined with the above, the execution order for a chain m1 → m2 → m3 (where each move's destination is the next move's source) is: m1 first, then m2, then m3.

**Convergent moves:** when two players move onto the same neutral tile simultaneously, the moves resolve in sequence under the priority order. The mover with more armies on their source tile captures the empty tile first; the second player's move then attacks the now-occupied tile as normal combat (see §7). The net effect: the tile goes to the player with the larger source army, with garrison equal to the difference between the two sent armies. If the two source armies are exactly equal, the input-order tiebreak applies, and the resulting garrison is 0 (defender's advantage in tied combat).

---

## 7. Combat

Combat occurs when a player moves onto a tile owned by an opponent (or a neutral city/neutral army tile). Neutral cities are garrisoned with 40–50 armies (see §2), so capturing one typically requires a source tile with at least garrison + 2 armies (garrison + 1 to overcome the defender, +1 for the army left behind).

**Resolution:** larger army wins. The attacking army is the amount sent from the source tile (`a - 1` for a normal move, `floor(a / 2)` for a split move). The defending army is whatever is currently on the target tile.

**If attacker wins:** the target tile's ownership changes to the attacker. The resulting army on the tile equals the attacker's sent army minus the defender's army. The source tile retains 1 army (normal move) or `ceil(a / 2)` army (split move).

**If defender wins:** tile ownership is unchanged. The defender's army is reduced by the attacker's sent army. The source tile retains its leftover army.

**If armies are equal:** the defender wins (defender's advantage). The tile stays with the defender at 0 army.

**Effective capture threshold:** to capture a tile with `d` defending armies using a normal move, you need at least `d + 2` armies on your source tile — `d + 1` to overcome the defender, plus 1 to leave behind. For a split move, you need at least `2 * (d + 1)` on the source tile.

**Multiple simultaneous attackers.** When two or more players attack the same tile in the same timestep, the moves resolve sequentially per the move priority order (§6). Each successive attacker sees the post-resolution state of the defending tile — which may have been weakened or captured by an earlier-resolving move.

---

## 8. Scoreboard

The scoreboard is always visible to all players, updated every timestep. It shows two values per player:

- **Total land count:** number of tiles owned.
- **Total army count:** sum of armies across all owned tiles.

The scoreboard is the primary information channel for opponents behind fog. Because it updates every timestep and reflects every aggregate change immediately, it enables strategic inferences (see §10).

The scoreboard is symmetric — opponents can make the same inferences about you.

---

## 9. Elimination, surrender, and game end

**Elimination:** capturing a player's general eliminates them from the game.

**Inheritance:** the capturing player inherits all of the eliminated player's territory — land, cities, and armies. Two important details:

- **The captured general becomes a city.** It functions identically to any other owned city: produces +1 army per turn, can be contested and captured by other players, etc. The army count on this tile is whatever was left after the capturing combat — the halving rule below does not apply to it (it has already absorbed the combat damage).
- **Inherited armies are halved (rounded upward).** Army counts on the defeated player's other tiles are halved when ownership transfers to the capturing player. A tile with 10 armies becomes 5 under the new owner; a tile with 11 becomes 6.

**Notification:** Player captures are announced globally as a system-message in the game chat.

* The message identifies both the eliminated player and the capturing player by username and color.
* E.g. "Player 1 (blue square) has captured Player 3 (green square)"

**Surrender / disconnect:** a player may surrender at any time. Surrender is not instant — there is a **25-turn countdown** (50 timesteps, 25 seconds at normal game speed; one full round) during which the surrendering player is still in the game and their general is still capturable. This countdown exists to protect opponents who were actively fighting the surrendering player: without it, a surrender would deny them the capture and inheritance they were about to earn.

The rule also applies to disconnects, except that disconnect uses a 1-timestep countdown — effectively immediate.

**During the countdown:** the surrendering player is marked as eliminated and can no longer issue moves, but their tiles remain under their ownership. Their general and cities continue to produce armies. Other players can attack and capture their tiles and general normally during this window.

After the countdown expires:

- All of the surrendering player's tiles convert to neutral, with army values unchanged.
- The general and cities become neutral cities, garrisoned with whatever army count they had on the final tick of the countdown.
- Non-structure tiles become neutral land with retained armies (see "neutral armies" in §2).

**Two ways the countdown can be cut short:**

- **Captured first.** If another player captures the surrendering player's general before the countdown expires, the normal capture-and-inheritance rules apply. No neutralization happens — the player is already captured, and the captor inherits everything per the rules above.
- **Game ends first.** If the surrender drops the game to a single remaining player (a common outcome of late-game surrender), the game ends immediately. No neutralization needed.

**Game end:** the game ends when only one general remains.

**Synthetic game-end conditions.** In rare situations where the normal one-general condition doesn't trigger, the game ends via one of two safety fallbacks:

- **All-AFK fallback.** If no moves are made by any player for 2000 consecutive timesteps (~17 minutes at normal speed), all remaining players except the strongest (by total army count, then tile count) are eliminated — the strongest is declared the winner.
- **Maximum game length.** Games are capped at 50000 timesteps (~7 hours at normal speed). At the cap, the same "kill all but the leader" fallback fires.

---

## 10. Immediate strategic implications

*The following are not comprehensive strategy — they are direct consequences of the mechanics above that are relevant to neural network design and general understanding of the game's strategic texture. This list is not meant to be comprehensive, but highlights several noteworthy items.*

**Permanent expansion.** Because movement always leaves at least 1 army behind (§6), every move forward is a commitment that extends your border and your vision. There is no "exploring and then leaving" — moving through an area means you now own every tile along the path and retain vision of all of them plus their Moore neighborhoods.

**Vision only shrinks from enemy action.** Your fog-of-war visibility boundary only ever expands or stays the same from your own actions. Vision loss is exclusively caused by enemy capture. Every tile you've ever moved through is still yours (unless an opponent has since taken it). The path history of your armies is literally written into the ownership map.

**City count inference.** During peacetime, army growth per turn = 1 (general) + number of cities. Observing an opponent's growth rate on the scoreboard reveals their city count. For example, if a player's army grows by 3 per turn during peacetime, they have 2 cities plus their general.

**Conflict detection.** If one player's land is shrinking while another's is growing at a similar rate, they're likely fighting each other.

**Relative strength assessment.** Army and land totals relative to the field indicate who is leading, trailing, or consolidating.

**Inferring the position of enemy generals.** The general placement rule during map generation (min distance of 10 between generals) defines a guaranteed exclusion zone around your own general where enemy generals cannot be, which reduces the search space when hunting for opponents. And this compounds dramatically: each additional general position that is discovered or inferred further narrows the possible locations of the remaining generals.

**Information-asymmetry.** Fog of war is the central information-asymmetry mechanic. It is why memory features matter for the neural network (tracking "what did I last see here, and how stale is that information") and why scoreboard-derived inferences are so valuable (they provide signal about opponents you can't directly see).

**Map navigation.**  The special visibility rules for structures have large implications for map navigation as well as attack and defense concerns. Unexplored structures in the fog may be capturable neutral cities. This means that what might appear to be an impassable mountain line dividing a region of the map is actually traversable through a neutral city. An illustrative example: *a general tucked into a back cave must explore the surrounding structures to ensure there isn't a hidden "backdoor."*

**The power of capturing.** There are two ways to gain land. (1) Traverse and capture neutral tiles one move at a time **or** (2) gain all land from a player by attacking and capturing their general tile. Capturing the general of a player with 100 owned tiles increases your land by +100 in a single move, while accomplishing that through neutral tiles would take 100 moves. In actuality, winning a fight may take many moves, but the core framing is very important and is a key part of the strategy. Killing other players is often the fastest way to grow stronger. And it's compounded by the fact that "other players" can be viewed as a "limited resource" that one must compete for. If player A captures player B, that general is gone (it's now a city), and no one else can capture that player and gain that entire set of tiles in one fell swoop.

