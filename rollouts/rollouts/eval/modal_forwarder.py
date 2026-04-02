from __future__ import annotations

import argparse
import asyncio
import signal

import modal


async def _run_forwarder(port: int) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_event.set)

    async with modal.forward(port) as tunnel:
        print(f"Modal forward active on port {port}: {tunnel.url}", flush=True)
        await stop_event.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forward a sandbox port through Modal.")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    asyncio.run(_run_forwarder(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
