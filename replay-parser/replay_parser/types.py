from dataclasses import dataclass


type PlayerIndex = int
type TileIndex = int
type Timestep = int
type PerspectiveIndex = int
type MoveRowIndex = int


@dataclass(frozen=True, slots=True)
class PerspectiveMetadata:
    player_id: PlayerIndex
    perspective_index: PerspectiveIndex
    rolling_1st_rate_at_game_time: float
    rolling_top3_rate_at_game_time: float
    prior_games_count_at_game_time: int
    stars_at_start: float
    placement: int
    elim_timestep: Timestep | None
