import logging
import os

from dotenv import load_dotenv

load_dotenv()


def cpu_count() -> int | None:
    cpu_count = os.getenv("NUM_CPUS", None)
    if cpu_count is not None and cpu_count.isdigit():
        if (os_cpu_count := os.cpu_count()) is not None and int(cpu_count) > os_cpu_count:
            logging.warning(f"NUM_CPUS={cpu_count} is greater than the number of CPUs available ({os_cpu_count}).")
        return int(cpu_count)
    else:
        return None
