# Cohort Scheduler

Stochastic greedy scheduling algorithm for matching people into cohorts based on availability.

## Installation

```bash
pip install git+https://github.com/yourusername/cohort-scheduler.git
```

Or clone and install locally:

```bash
git clone https://github.com/yourusername/cohort-scheduler.git
cd cohort-scheduler
pip install -e .
```

## Usage

```python
from cohort_scheduler import Person, schedule, format_time_range, parse_interval_string

# Create people with availability
people = [
    Person(
        id="1",
        name="Alice",
        intervals=parse_interval_string("M09:00 M11:00, W14:00 W16:00")
    ),
    Person(
        id="2",
        name="Bob",
        intervals=parse_interval_string("M10:00 M12:00, W14:00 W17:00")
    ),
    Person(
        id="3",
        name="Charlie",
        intervals=parse_interval_string("M09:00 M12:00, W15:00 W17:00")
    ),
    Person(
        id="4",
        name="Diana",
        intervals=parse_interval_string("M10:00 M11:00, W14:00 W16:00")
    ),
]

# Run scheduling
result = schedule(
    people=people,
    meeting_length=60,      # 1 hour meetings
    min_people=2,           # At least 2 per group
    max_people=4,           # At most 4 per group
    num_iterations=1000,    # Try 1000 random variations
)

# Print results
print(f"Scheduled {result.score}/{len(people)} people into {len(result.groups)} groups")

for group in result.groups:
    print(f"\n{group.name}:")
    print(f"  Members: {[p.name for p in group.people]}")
    if group.selected_time:
        print(f"  Meeting: {format_time_range(*group.selected_time)}")

if result.unassigned:
    print(f"\nUnassigned: {[p.name for p in result.unassigned]}")
```

## Interval Format

Availability is specified as comma-separated intervals:
- Day codes: M=Monday, T=Tuesday, W=Wednesday, R=Thursday, F=Friday, S=Saturday, U=Sunday
- Format: `{day}{HH:MM} {day}{HH:MM}` (start and end)
- Example: `"M09:00 M11:00, W14:00 W16:00"` = Monday 9-11am and Wednesday 2-4pm

## API

### `schedule(people, **options) -> SchedulingResult`

Main function. Options:
- `meeting_length`: Duration in minutes (default: 60)
- `min_people`: Minimum per group (default: 4)
- `max_people`: Maximum per group (default: 8)
- `num_iterations`: Random iterations to try (default: 10000)
- `time_increment`: Time slot granularity in minutes (default: 30)
- `randomness`: 0-1, how much randomness (default: 0.5)
- `use_if_needed`: Include "if needed" availability (default: True)
- `balance`: Balance group sizes after scheduling (default: True)
- `facilitator_ids`: Set of facilitator IDs (optional)
- `facilitator_max_cohorts`: Dict of facilitator -> max groups (optional)

### Other Functions

- `parse_interval_string(s)`: Parse "M09:00 M11:00" format to intervals
- `format_time_range(start, end)`: Format as "Monday 09:00 - 11:00"
- `is_group_valid(people, meeting_length, ...)`: Check if group has valid meeting time
- `find_meeting_times(people, meeting_length, ...)`: Get all valid meeting times
- `balance_groups(groups, meeting_length, ...)`: Balance group sizes

## License

MIT
