import pandas as pd
from pydantic import BaseModel, Field


class EngagementInterval(BaseModel):
    unit: str = Field(..., description="The unit of time for the interval")
    quantity: int = Field(..., description="The quantity of units for the interval")


class SkillSet(BaseModel):
    leadership: float = Field(..., description="The leadership skill level")
    practices: float = Field(..., description="The practices skill level")
    technical: float = Field(..., description="The technical skill level")

    class Config:
        frozen = True


class Coach(BaseModel):
    name: str = Field(..., description="The name of the coach")
    weekly_availability: float = Field(..., description="The number of hours the coach is available per week")
    skill_set: SkillSet = Field(..., description="The skill set of the coach")

    class Config:
        frozen = True


class Area(BaseModel):
    name: str = Field(..., description="The name of the area")

    class Config:
        frozen = True


class AreaNeed(BaseModel):
    area: Area = Field(..., description="The area")
    skill_set: SkillSet = Field(..., description="The skill set needed")

    class Config:
        frozen = True


class CoachFormationModel(BaseModel):
    name: str = Field(..., description="The name of the coach formation")
    description: str = Field(..., description="A description of the coach formation")
    number_of_coaches: int = Field(..., description="The number of coaches in the formation")
    engagement_level: float = Field(..., description="The engagement capacity demands on each coach in the formation")


class ServiceModel(BaseModel):
    name: str = Field(..., description="The name of the service")
    objective: str = Field(..., description="The objective of the service")
    interval: EngagementInterval = Field(..., description="The interval of the service")
    formation: CoachFormationModel = Field(..., description="The formation of coaches that deliver the service")


class ServiceAssignment(BaseModel):
    model: ServiceModel = Field(..., description="The model of the service")
    coaches: list[Coach] = Field(..., description="The coaches that deliver the service")
    date_range: object = Field(..., description="The date range of the service")
    area: Area = Field(..., description="The area of the service")


class CoachHistory(BaseModel):
    coach: Coach = Field(..., description="The coach")
    service_assignment: ServiceAssignment = Field(..., description="The service assignment")


class Backlog(BaseModel):
    area_needs: list[AreaNeed] = Field(..., description="The backlog of area needs")


class Schedule(BaseModel):
    start_date: object = Field(..., description="The start date of the schedule")
    end_date: object = Field(..., description="The end date of the schedule")
    assignments: list[ServiceAssignment] = Field(..., description="The schedule of service assignments")

    @property
    def scheduled_areas(self) -> list[Area]:
        return list(set([assignment.area for assignment in self.assignments]))

    @property
    def scheduled_coaches(self) -> list[Coach]:
        return list(set([coach for assignment in self.assignments for coach in assignment.coaches]))

    @property
    def total_business_days(self) -> int:
        return len(pd.bdate_range(start=self.start_date, end=self.end_date, freq="C"))

    @property
    def total_assigned_days(self) -> int:
        return len(set([date for assignment in self.assignments for date in assignment.date_range]))

    @property
    def duty_cycle(self) -> float:
        return self.total_assigned_days / self.total_business_days

    def add_assignment(self, assignment: ServiceAssignment):
        self.assignments.append(assignment)
