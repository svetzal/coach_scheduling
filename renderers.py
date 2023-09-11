from models import Schedule


class PlainTextCalendarRenderer:
    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    def main_calendar(self) -> str:
        out = []
        for area in self.schedule.areas:
            out.append(f"Area: {area.name}")
            for assignment in self.schedule.assignments:
                if assignment.area == area:
                    out.append(f"  {assignment.model.name}: {assignment.date_range[0]} - {assignment.date_range[-1]}")
        return "\n".join(out)

    def coach_calendar(self) -> str:
        out = []
        for coach in self.schedule.coaches:
            out.append(f"Coach: {coach.name}")
            for assignment in self.schedule.assignments:
                if coach in assignment.coaches:
                    paired_coach = [c for c in assignment.coaches if c != coach][0]
                    out.append(
                        f"  {assignment.model.name} with {paired_coach.name} in {assignment.area.name}: {assignment.date_range[0]} - {assignment.date_range[-1]}")
        return "\n".join(out)

    def area_calendar(self) -> str:
        out = []
        for area in self.schedule.areas:
            out.append(f"Area: {area.name}")
            for assignment in self.schedule.assignments:
                if assignment.area == area:
                    out.append(f"  {assignment.model.name} with {', '.join([c.name for c in assignment.coaches])}: {assignment.date_range[0]} - {assignment.date_range[-1]}")
        return "\n".join(out)