echo "=== FFA replays: date range per version ==="

sqlite3 -column -header data/generals.sqlite "
       SELECT
              version,
              COUNT(*) AS games,
              strftime('%Y-%m-%dT%H:%M:%SZ', MIN(started)/1000, 'unixepoch') AS first_seen,
              strftime('%Y-%m-%dT%H:%M:%SZ', MAX(started)/1000, 'unixepoch') AS last_seen
       FROM replays r
       WHERE (r.ladder_id = 'ffa' AND r.player_count BETWEEN 4 AND 8)
       GROUP BY version
       ORDER BY version;"
