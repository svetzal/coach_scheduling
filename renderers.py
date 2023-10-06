from models import Schedule


class PlainTextCalendarRenderer:
    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    def main_calendar(self) -> str:
        out = []
        for area in self.schedule.scheduled_areas:
            out.append(f"Area: {area.name}")
            for assignment in self.schedule.assignments:
                if assignment.area == area:
                    out.append(f"  {assignment.model.name}: {assignment.date_range[0]} - {assignment.date_range[-1]}")
        return "\n".join(out)

    def coach_calendar(self) -> str:
        out = []
        for coach in self.schedule.scheduled_coaches:
            out.append(f"Coach: {coach.name}")
            for assignment in self.schedule.assignments:
                if coach in assignment.coaches:
                    paired_coach = [c for c in assignment.coaches if c != coach][0]
                    out.append(
                        f"  {assignment.model.name} with {paired_coach.name} in {assignment.area.name}: {assignment.date_range[0]} - {assignment.date_range[-1]}")
        return "\n".join(out)

    def area_calendar(self) -> str:
        out = []
        for area in self.schedule.scheduled_areas:
            out.append(f"Area: {area.name}")
            for assignment in self.schedule.assignments:
                if assignment.area == area:
                    out.append(
                        f"  {assignment.model.name} with {', '.join([c.name for c in assignment.coaches])}: {assignment.date_range[0]} - {assignment.date_range[-1]}")
        return "\n".join(out)


class TailwindHtmlScheduleRender:
    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    def preamble(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Agile Coach Schedule</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.16/tailwind.min.css">
</head>
<body>
<div class="container mx-auto p-4">
"""

    def postamble(self) -> str:
        return """
</div>
</body>
</html>
"""

    def main_calendar(self) -> str:
        out = []
        for area in self.schedule.scheduled_areas:
            out.append(f"<h2 class='text-2xl font-bold mb-2'>{area.name}</h2>")
            for assignment in self.schedule.assignments:
                if assignment.area == area:
                    out.append(
                        f"<p class='text-lg font-semibold'>{assignment.model.name}: {assignment.date_range[0]} - {assignment.date_range[-1]}</p>")
        return "\n".join(out)

    def coach_calendar(self) -> str:
        out = []
        for coach in self.schedule.scheduled_coaches:
            out.append(f"<h2 class='text-2xl font-bold mb-2'>{coach.name}</h2>")
            for assignment in self.schedule.assignments:
                if coach in assignment.coaches:
                    paired_coach = [c for c in assignment.coaches if c != coach][0]
                    out.append(
                        f"<p class='text-lg font-semibold'>{assignment.model.name} with {paired_coach.name} in {assignment.area.name}: {assignment.date_range[0]} - {assignment.date_range[-1]}</p>")
        return "\n".join(out)

    def area_calendar(self) -> str:
        out = []
        for area in self.schedule.scheduled_areas:
            out.append(f"<h2 class='text-2xl font-bold mb-2'>{area.name}</h2>")
            for assignment in self.schedule.assignments:
                if assignment.area == area:
                    out.append(
                        f"<p class='text-lg font-semibold'>{assignment.model.name} with {', '.join([c.name for c in assignment.coaches])}: {assignment.date_range[0]} - {assignment.date_range[-1]}</p>")
        return "\n".join(out)

    def render(self):
        return self.preamble() + self.main_calendar() + self.coach_calendar() + self.area_calendar() + self.postamble()
