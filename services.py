import pandas as pd

from models import EngagementInterval, Coach, Area, CoachFormationModel, ServiceModel, ServiceAssignment, Schedule
from renderers import PlainTextCalendarRenderer

six_week_interval = EngagementInterval(
    unit="week",
    quantity=6
)

pair_formation_model = CoachFormationModel(
    name="pair",
    description="Two coaches work together to deliver the service",
    number_of_coaches=2,
    engagement_level=0.8
)

one_on_one_formation_model = CoachFormationModel(
    name="one-on-one",
    description="One coach works with one client to deliver the service",
    number_of_coaches=1,
    engagement_level=0.1
)

anchored_formation_model = CoachFormationModel(
    name="anchored",
    description="One coach performs general agile coaching activities within one area",
    number_of_coaches=1,
    engagement_level=0.8
)

dojo = ServiceModel(
    name="Dojo",
    objective="Team annealing",
    interval=six_week_interval,
    formation=pair_formation_model
)

tour = ServiceModel(
    name="Tour",
    objective="Team annealing",
    interval=six_week_interval,
    formation=pair_formation_model
)

one_on_one = ServiceModel(
    name="One-on-one",
    objective="Adoption of agile thinking",
    interval=six_week_interval,
    formation=one_on_one_formation_model
)

coach_1 = Coach(name="Coach 1", weekly_availability=30)
coach_2 = Coach(name="Coach 2", weekly_availability=30)
coach_3 = Coach(name="Coach 3", weekly_availability=30)
coach_4 = Coach(name="Coach 4", weekly_availability=30)
coaches = [coach_1, coach_2, coach_3, coach_4]
pair_1 = [coach_1, coach_2]
pair_2 = [coach_3, coach_4]
pairs = [pair_1, pair_2]

areas = [Area(name=f"Area {i}") for i in range(1, 5)]

schedule = Schedule(
    start_date=pd.to_datetime("2023-11-01"),
    end_date=pd.to_datetime("2024-10-31"),
    assignments=[]
)

current_date = schedule.start_date
tour_length = pd.DateOffset(weeks=tour.interval.quantity)

area_index = 0
while current_date <= schedule.end_date - tour_length:
    for pair in pairs:
        schedule.add_assignment(
            ServiceAssignment(
                model=tour,
                coaches=pair,
                date_range=pd.bdate_range(start=current_date, end=current_date + tour_length, freq="C"),
                area=areas[area_index % len(areas)]
            )
        )
        area_index += 1
    current_date += tour_length

renderer = PlainTextCalendarRenderer(schedule)
print("\nCentral Calendar\n\n", renderer.main_calendar())
print("\nCoach Calendars\n\n", renderer.coach_calendar())
print("\nArea Calendars\n\n", renderer.area_calendar())

print(f"\nCalendar Total business days: {schedule.business_days}")
print(f"Calendar assigned days: {schedule.assigned_days}")
print(f"Duty cycle: {schedule.duty_cycle}")

unserviced_areas = [area for area in areas if area not in schedule.areas]
print(f"\nUnserviced areas: {unserviced_areas}")

unused_coaches = [coach for coach in coaches if coach not in schedule.coaches]
print(f"Unused coaches: {unused_coaches}")
