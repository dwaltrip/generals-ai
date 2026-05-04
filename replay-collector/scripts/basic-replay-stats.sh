echo "=== Totals ==="

sqlite3 --column --header data/generals.sqlite "
       SELECT COUNT(*) AS total,
              SUM(raw IS NOT NULL) AS with_raw
              FROM replays;"

echo "" && echo "=== By month (raw fetched) ==="

sqlite3 -column -header data/generals.sqlite "
       SELECT
              strftime('%Y-%m', started/1000, 'unixepoch') AS month,
              COUNT(*) AS n
       FROM replays
       WHERE raw IS NOT NULL GROUP BY month ORDER BY month;"

echo "" && echo "=== By game version ==="

sqlite3 -column -header data/generals.sqlite "
       SELECT version, COUNT(*) AS n
       FROM replays WHERE raw IS NOT NULL
       GROUP BY version ORDER BY version;"
       
echo "" && echo "=== By player count ==="
      
sqlite3 -column -header data/generals.sqlite "
       SELECT player_count, COUNT(*) AS n FROM replays
       WHERE raw IS NOT NULL
       GROUP BY player_count ORDER BY player_count;"

echo "" && echo "=== By type ==="

sqlite3 -column -header data/generals.sqlite "
       SELECT type, COUNT(*) AS n
       FROM replays WHERE raw IS NOT NULL
       GROUP BY type ORDER BY n DESC;"
