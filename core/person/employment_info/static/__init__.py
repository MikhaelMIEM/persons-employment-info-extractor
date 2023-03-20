from importlib import resources
from typing import Iterator


def read_raw_job_titles() -> Iterator[str]:
    with resources.open_text('core.person.employment_info.static', 'job_titles_dict.txt') as dict_file:
        for line in dict_file:
            if line:
                yield line.strip()
