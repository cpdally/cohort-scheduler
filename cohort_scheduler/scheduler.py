"""
Cohort Scheduling Algorithm

Stochastic greedy scheduling for matching people into cohorts based on availability.
Runs many iterations of greedy assignment, keeps best solution.
"""

import random
from dataclasses import dataclass, field
from typing import Optional


DAY_MAP = {'M': 0, 'T': 1, 'W': 2, 'R': 3, 'F': 4, 'S': 5, 'U': 6}
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MINUTES_PER_WEEK = 7 * 24 * 60


@dataclass
class Person:
    """Represents a person for scheduling."""
    id: str
    name: str
    intervals: list  # List of (start_minutes, end_minutes) tuples
    if_needed_intervals: list = field(default_factory=list)
    timezone: str = "UTC"


@dataclass
class Group:
    """Represents a scheduled group/cohort."""
    id: str
    name: str
    people: list
    facilitator_id: Optional[str] = None
    selected_time: Optional[tuple] = None


@dataclass
class SchedulingResult:
    """Result of the scheduling algorithm."""
    groups: list
    unassigned: list
    score: int
    iterations_run: int
    best_iteration: int


def parse_interval_string(interval_str: str) -> list:
    """
    Parse availability string into intervals.
    Format: "M09:00 M10:00, T14:00 T15:00"
    Returns list of (start_minutes, end_minutes) tuples.
    """
    if not interval_str:
        return []

    intervals = []
    for part in interval_str.split(','):
        trimmed = part.strip()
        if not trimmed:
            continue

        tokens = trimmed.split()
        if len(tokens) < 2:
            continue

        start_token, end_token = tokens[0], tokens[1]

        start_day = DAY_MAP.get(start_token[0], 0)
        start_time = start_token[1:].split(':')
        start_minutes = start_day * 24 * 60 + int(start_time[0]) * 60 + int(start_time[1])

        end_day = DAY_MAP.get(end_token[0], 0)
        end_time = end_token[1:].split(':')
        end_minutes = end_day * 24 * 60 + int(end_time[0]) * 60 + int(end_time[1])

        if end_minutes <= start_minutes:
            end_minutes += MINUTES_PER_WEEK

        intervals.append((start_minutes, end_minutes))

    return intervals


def format_time_range(start_minutes: int, end_minutes: int) -> str:
    """Format time range as human-readable string."""
    start_day = start_minutes // (24 * 60)
    start_hour = (start_minutes % (24 * 60)) // 60
    start_min = start_minutes % 60

    end_day = end_minutes // (24 * 60)
    end_hour = (end_minutes % (24 * 60)) // 60
    end_min = end_minutes % 60

    fmt = lambda h, m: f"{h:02d}:{m:02d}"

    if start_day == end_day:
        return f"{DAY_NAMES[start_day % 7]} {fmt(start_hour, start_min)} - {fmt(end_hour, end_min)}"
    return f"{DAY_NAMES[start_day % 7]} {fmt(start_hour, start_min)} - {DAY_NAMES[end_day % 7]} {fmt(end_hour, end_min)}"


def _calculate_total_available_time(person: Person) -> int:
    total = sum(end - start for start, end in person.intervals)
    total += sum(end - start for start, end in person.if_needed_intervals)
    return total


def _is_available_at_time(person: Person, check_time: int, use_if_needed: bool) -> bool:
    if any(start <= check_time < end for start, end in person.intervals):
        return True
    if use_if_needed:
        return any(start <= check_time < end for start, end in person.if_needed_intervals)
    return False


def is_group_valid(
    people: list,
    meeting_length: int,
    time_increment: int = 30,
    use_if_needed: bool = True,
    facilitator_ids: set = None
) -> bool:
    """Check if a group of people has at least one valid meeting time."""
    if not people:
        return True

    if facilitator_ids:
        facilitators_in_group = [p for p in people if p.id in facilitator_ids]
        if len(facilitators_in_group) != 1:
            return False

    for time_in_minutes in range(0, MINUTES_PER_WEEK, time_increment):
        block_is_valid = True
        for offset in range(0, meeting_length, time_increment):
            check_time = time_in_minutes + offset
            if not all(_is_available_at_time(p, check_time, use_if_needed) for p in people):
                block_is_valid = False
                break
        if block_is_valid:
            return True
    return False


def find_meeting_times(
    people: list,
    meeting_length: int,
    time_increment: int = 30,
    use_if_needed: bool = True
) -> list:
    """Find all possible meeting time slots for a group of people."""
    options = []
    for time_in_minutes in range(0, MINUTES_PER_WEEK, time_increment):
        block_is_valid = True
        for offset in range(0, meeting_length, time_increment):
            check_time = time_in_minutes + offset
            if not all(_is_available_at_time(p, check_time, use_if_needed) for p in people):
                block_is_valid = False
                break
        if block_is_valid:
            options.append((time_in_minutes, time_in_minutes + meeting_length))
    return options


def _run_greedy_iteration(
    people, meeting_length, min_people, max_people, max_groups,
    time_increment, randomness, use_if_needed, facilitator_ids, facilitator_max_cohorts
) -> list:
    if facilitator_ids and len(facilitator_ids) > 0:
        facilitators = [p for p in people if p.id in facilitator_ids]
        non_facilitators = [p for p in people if p.id not in facilitator_ids]

        non_facilitators_sorted = sorted(non_facilitators, key=lambda p: (
            _calculate_total_available_time(p) * (1.0 - randomness * 0.1 + random.random() * randomness * 0.2)
        ))

        new_groups = []
        facilitator_assignments = {f.id: 0 for f in facilitators}

        for person in non_facilitators_sorted:
            placed = False

            valid_group_indices = []
            for i, group in enumerate(new_groups):
                if len(group.people) < max_people:
                    test_people = group.people + [person]
                    if is_group_valid(test_people, meeting_length, time_increment, use_if_needed, facilitator_ids):
                        valid_group_indices.append(i)

            if valid_group_indices:
                if randomness == 0 or random.random() > randomness:
                    selected_index = valid_group_indices[0]
                else:
                    selected_index = random.choice(valid_group_indices)
                new_groups[selected_index].people.append(person)
                placed = True

            if not placed and len(new_groups) < max_groups:
                for facilitator in facilitators:
                    max_cohorts = facilitator_max_cohorts.get(facilitator.id, 1) if facilitator_max_cohorts else 1
                    current_count = facilitator_assignments[facilitator.id]

                    if current_count < max_cohorts:
                        test_people = [facilitator, person]
                        if is_group_valid(test_people, meeting_length, time_increment, use_if_needed, facilitator_ids):
                            new_groups.append(Group(
                                id=f"group-{len(new_groups)}",
                                name=f"Group {len(new_groups) + 1}",
                                people=[facilitator, person],
                                facilitator_id=facilitator.id
                            ))
                            facilitator_assignments[facilitator.id] = current_count + 1
                            placed = True
                            break

        return [g for g in new_groups if len(g.people) >= min_people]

    else:
        people_sorted = sorted(people, key=lambda p: (
            _calculate_total_available_time(p) * (1.0 - randomness * 0.1 + random.random() * randomness * 0.2)
        ))

        new_groups = []

        for person in people_sorted:
            placed = False

            valid_group_indices = []
            for i, group in enumerate(new_groups):
                if len(group.people) < max_people:
                    test_people = group.people + [person]
                    if is_group_valid(test_people, meeting_length, time_increment, use_if_needed):
                        valid_group_indices.append(i)

            if valid_group_indices:
                if randomness == 0 or random.random() > randomness:
                    selected_index = valid_group_indices[0]
                else:
                    selected_index = random.choice(valid_group_indices)
                new_groups[selected_index].people.append(person)
                placed = True

            if not placed and len(new_groups) < max_groups:
                new_groups.append(Group(
                    id=f"group-{len(new_groups)}",
                    name=f"Group {len(new_groups) + 1}",
                    people=[person]
                ))

        return [g for g in new_groups if len(g.people) >= min_people]


def balance_groups(
    groups: list,
    meeting_length: int,
    time_increment: int = 30,
    use_if_needed: bool = True
) -> int:
    """Balance group sizes by moving people from larger to smaller groups."""
    if len(groups) < 2:
        return 0

    move_count = 0
    improved = True

    while improved:
        improved = False
        groups.sort(key=lambda g: len(g.people), reverse=True)

        if len(groups[0].people) - len(groups[-1].people) <= 1:
            break

        found_move = False
        for source_idx in range(len(groups)):
            if found_move:
                break
            source_group = groups[source_idx]

            for target_idx in range(len(groups) - 1, source_idx, -1):
                if found_move:
                    break
                target_group = groups[target_idx]

                if len(source_group.people) <= len(target_group.people):
                    continue

                for i, person in enumerate(source_group.people):
                    test_people = target_group.people + [person]
                    if is_group_valid(test_people, meeting_length, time_increment, use_if_needed):
                        source_group.people.pop(i)
                        target_group.people.append(person)
                        move_count += 1
                        improved = True
                        found_move = True
                        break

        if not found_move:
            break

    return move_count


def schedule(
    people: list,
    meeting_length: int = 60,
    min_people: int = 4,
    max_people: int = 8,
    max_groups: int = 999,
    num_iterations: int = 10000,
    time_increment: int = 30,
    randomness: float = 0.5,
    use_if_needed: bool = True,
    balance: bool = True,
    facilitator_ids: set = None,
    facilitator_max_cohorts: dict = None,
    progress_callback = None
) -> SchedulingResult:
    """
    Run the stochastic greedy scheduling algorithm.

    Args:
        people: List of Person objects to schedule
        meeting_length: Duration of meeting in minutes (default 60)
        min_people: Minimum people per cohort (default 4)
        max_people: Maximum people per cohort (default 8)
        max_groups: Maximum number of groups to create (default 999)
        num_iterations: Number of random iterations to try (default 10000)
        time_increment: Granularity of time slots in minutes (default 30)
        randomness: How much randomness to add (0-1, default 0.5)
        use_if_needed: Whether to consider "if needed" availability (default True)
        balance: Whether to balance group sizes after scheduling (default True)
        facilitator_ids: Set of person IDs who are facilitators (or None)
        facilitator_max_cohorts: Dict of facilitator_id -> max cohorts they can lead
        progress_callback: Optional fn(iteration, num_iterations, best_score, total_people)

    Returns:
        SchedulingResult with groups, unassigned people, and stats
    """
    if not people:
        return SchedulingResult(groups=[], unassigned=[], score=0, iterations_run=0, best_iteration=-1)

    best_solution = None
    best_score = -1
    best_iteration = -1
    total_people = len(people)

    for iteration in range(num_iterations):
        solution = _run_greedy_iteration(
            people, meeting_length, min_people, max_people, max_groups,
            time_increment, randomness, use_if_needed, facilitator_ids, facilitator_max_cohorts
        )

        score = sum(len(g.people) for g in solution)

        if score > best_score:
            best_score = score
            best_solution = solution
            best_iteration = iteration
            if best_score == total_people:
                break

        if progress_callback and iteration % 100 == 0:
            progress_callback(iteration, num_iterations, best_score, total_people)

    if balance and best_solution and len(best_solution) >= 2:
        balance_groups(best_solution, meeting_length, time_increment, use_if_needed)

    if best_solution:
        for group in best_solution:
            options = find_meeting_times(group.people, meeting_length, time_increment, use_if_needed)
            if options:
                group.selected_time = options[0]
        assigned_ids = {p.id for g in best_solution for p in g.people}
        unassigned = [p for p in people if p.id not in assigned_ids]
    else:
        best_solution = []
        unassigned = list(people)

    return SchedulingResult(
        groups=best_solution,
        unassigned=unassigned,
        score=best_score if best_score >= 0 else 0,
        iterations_run=best_iteration + 1 if best_iteration >= 0 else 0,
        best_iteration=best_iteration
    )
