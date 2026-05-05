# Generals.io: Game Mechanics Reference

**Date:** 2026.05.03

**Purpose:** Canonical reference for the core mechanics of generals.io, written to support neural network architecture and replay parser design. Other project docs assume familiarity with this material.

**Scope:** This document covers the standard FFA and 1v1 game mode. It does not cover team modes, custom games, special events, or other variants.

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

**Simultaneous execution:** all players' moves for a given timestep are submitted simultaneously, but when moves interact, they are resolved in a deterministic priority order.

**The "inward-first" rule:** for any pair of moves m1 and m2, if m1's destination is m2's source tile, then m1 executes before m2. Intuitively, a tile must face an incoming move before it can execute its own outgoing move. This has two important consequences:

- **Defending/chasing is favored.** If m1 attacks the tile that m2 is trying to move away from, m1 lands first. m2's source tile absorbs the hit — its army is reduced (or the tile is captured entirely) — before m2's outgoing move resolves. If m1 captures the tile, m2's move is canceled.
- **The rule chains.** If m3's source tile is m2's destination, then m2 executes before m3 by the same logic. Combined with the above, the execution order for a chain m1 → m2 → m3 (where each move's destination is the next move's source) is: m1 first, then m2, then m3.

**Convergent moves:** when two players move onto the same neutral tile simultaneously, it is resolved as combat between the two arriving armies. The larger army takes the tile with the difference as its garrison.*

---

## 7. Combat

Combat occurs when a player moves onto a tile owned by an opponent (or a neutral city/neutral army tile). Neutral cities are garrisoned with 40–50 armies (see §2), so capturing one typically requires a source tile with at least garrison + 2 armies (garrison + 1 to overcome the defender, +1 for the army left behind).

**Resolution:** larger army wins. The attacking army is the amount sent from the source tile (`a - 1` for a normal move, `floor(a / 2)` for a split move). The defending army is whatever is currently on the target tile.

**If attacker wins:** the target tile's ownership changes to the attacker. The resulting army on the tile equals the attacker's sent army minus the defender's army. The source tile retains 1 army (normal move) or `ceil(a / 2)` army (split move).

**If defender wins:** tile ownership is unchanged. The defender's army is reduced by the attacker's sent army. The source tile retains its leftover army.

**If armies are equal:** the defender wins (defender's advantage). The tile stays with the defender at 0 army.

**Effective capture threshold:** to capture a tile with `d` defending armies using a normal move, you need at least `d + 2` armies on your source tile — `d + 1` to overcome the defender, plus 1 to leave behind. For a split move, you need at least `2 * (d + 1)` on the source tile.

---

## 8. Scoreboard

The scoreboard is always visible to all players, updated every timestep. It shows two values per player:

- **Total land count:** number of tiles owned.
- **Total army count:** sum of armies across all owned tiles.

The scoreboard is the primary information channel for opponents behind fog. Because it updates every timestep and reflects every aggregate change immediately, it enables strategic inferences (see §10).

The scoreboard is symmetric — opponents can make the same inferences about you.

---

## 9. Elimination and inheritance

**Elimination:** capturing a player's general eliminates them from the game.

**Inheritance:** the capturing player inherits all of the eliminated player's territory — land, cities, and armies. Two important details:

- **The captured general becomes a city.** It functions identically to any other owned city: produces +1 army per turn, can be contested and captured by other players, etc.
- **Inherited armies are halved.** All army counts on the defeated player's tiles are divided by 2 (rounded down) when ownership transfers to the capturing player. A tile that had 10 armies under the defeated player will have 5 armies under the new owner.

**Notification:** Player captures are announced globally as a system-message in the game chat.

* The message identifies both the eliminated player and the capturing player by username and color.
* E.g. "Player 1 (blue square) has captured Player 3 (green square)"

**Game end:** the game ends when only one general remains.

**Surrender / disconnect:** a player may surrender at any time. Surrender is not instant — there is a countdown period* during which the surrendering player is still in the game and their general is still capturable. This countdown exists to protect opponents who were actively fighting the surrendering player: without it, a surrender would deny them the capture and inheritance they were about to earn. The rule also applies to disconnects.

After the countdown expires:

- All of the surrendering player's tiles convert to neutral, with army values unchanged.
- The general and cities become neutral cities, garrisoned with whatever army count they had on the final tick of the countdown.
- Non-structure tiles become neutral land with retained armies (see "neutral armies" in §2).

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

---

## 11. Ambiguities / needs confirmation

The following details are unconfirmed and should be tested and resolved when building the replay parser or simulator.

1. **Convergent moves and ties.** When two players move onto the same neutral tile simultaneously and their armies differ by exactly 1, does the larger army capture the tile with 1 remaining? Or is it treated as effectively a tie? (Referenced in §6.)
2. **Two attackers on one defender.** When two players simultaneously attack a tile owned by a third player, the exact resolution order is unknown. Possibilities include: one attack is prioritized by player index, larger army moves first, or something else entirely.
3. **Production timing on capture.** If a city is captured on the same timestep as a turn boundary, does it produce for the new owner that turn, or starting the next turn? Same question for the land tick — do tiles captured on the exact timestep of a land tick receive the +1? Believed to be "next tick" but not confirmed.
4. **Surrender countdown duration.** The surrender countdown is believed to be approximately 10–15 turns but the exact value is unconfirmed.
4. **Disconnect trigger:** When is a player considered disconnected and how is it represented in the replay data? Is that the moment the "surrender countdown" starts? For the 2nd question, the answer is believed to be "yes". But both questions need confirmation.
