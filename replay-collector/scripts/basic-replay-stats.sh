echo ""
echo "=== Totals ==="

sqlite3 --column --header data/generals.sqlite "
       SELECT
              COUNT(*) AS total,
              SUM(wire_data IS NOT NULL) AS with_wire_data
       FROM replays
       WHERE ladder_id = 'ffa';"

echo "" && echo "=== By month (raw fetched) ==="

sqlite3 -column -header data/generals.sqlite "
       SELECT
              strftime('%Y-%m', started/1000, 'unixepoch') AS month,
              COUNT(*) AS n
       FROM replays
       WHERE wire_data IS NOT NULL AND ladder_id = 'ffa'
       GROUP BY month
       ORDER BY month;"

echo "" && echo "=== By game version ==="

sqlite3 -column -header data/generals.sqlite "
       SELECT version, COUNT(*) AS n
       FROM replays
       WHERE wire_data IS NOT NULL AND ladder_id = 'ffa'
       GROUP BY version
       ORDER BY version;"
       
echo "" && echo "=== By player count ==="
      
sqlite3 -column -header data/generals.sqlite "
       SELECT player_count, COUNT(*) AS n FROM replays
       WHERE wire_data IS NOT NULL AND ladder_id = 'ffa'
       GROUP BY player_count
       ORDER BY player_count;"
