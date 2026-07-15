import asyncio
import logging
import os

from webagent.agent import run_task


def _configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


async def _main() -> None:
    result = await run_task(
        # task="Find and click the link to the Python documentation, then report the URL of the page you land on.",
        task="Find all open remote job listings on the Python.org website including links to the postings, and return them as a JSON array of objects with 'title' and 'url' fields.",
        url="https://www.python.org",
        output_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string", "format": "uri"}
                },
                "required": ["title", "url"]
            }
        },
    )
    print(result)


def main() -> None:
    _configure_logging()
    asyncio.run(_main())


if __name__ == "__main__":
    main()
