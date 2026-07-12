import asyncio
import logging
import os

from webagent.agent import run_task


def _configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


async def _main() -> None:
    answer = await run_task(
        # task="Find and click the link to the Python documentation, then report the URL of the page you land on.",
        task="How can I sign up for the python newsletter via RSS feed?",
        url="https://www.python.org",
    )
    print(answer)


def main() -> None:
    _configure_logging()
    asyncio.run(_main())


if __name__ == "__main__":
    main()
