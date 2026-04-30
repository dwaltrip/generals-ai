import sys

from replay_collector.fetcher import fetch_replay


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m replay_collector <replay_id>", file=sys.stderr)
        sys.exit(2)

    replay_id = sys.argv[1]
    arr = fetch_replay(replay_id)
    version, rid, w, h, usernames = arr[0], arr[1], arr[2], arr[3], arr[4]
    moves = arr[10]
    last_turn = moves[-1][4] if moves else 0

    print(f"id={rid} v{version} {w}x{h} players={len(usernames)} moves={len(moves)} last_turn={last_turn}")
    print(f"usernames: {usernames}")


if __name__ == "__main__":
    main()
