"""Cohort Scheduler - Stochastic greedy scheduling for matching people into cohorts."""

from .scheduler import (
    Person,
    Group,
    SchedulingResult,
    parse_interval_string,
    format_time_range,
    is_group_valid,
    find_meeting_times,
    balance_groups,
    schedule,
)

__version__ = "0.1.0"
__all__ = [
    "Person",
    "Group",
    "SchedulingResult",
    "parse_interval_string",
    "format_time_range",
    "is_group_valid",
    "find_meeting_times",
    "balance_groups",
    "schedule",
]
