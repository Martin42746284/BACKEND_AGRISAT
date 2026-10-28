import pytest
from pydantic import ValidationError

from app.schemas import BorneGpsCreate, BorneGpsUpdate


def test_borne_create_trims_name_and_accepts_valid_coordinates() -> None:
    borne = BorneGpsCreate(name="  Borne nord  ", lat=-18.908168717882273, lng=47.53105912163172)

    assert borne.name == "Borne nord"
    assert borne.lat == pytest.approx(-18.908168717882273)
    assert borne.lng == pytest.approx(47.53105912163172)


@pytest.mark.parametrize("name", ["", "   ", "\n\t"])
def test_borne_create_rejects_blank_name(name: str) -> None:
    with pytest.raises(ValidationError):
        BorneGpsCreate(name=name, lat=-18.9, lng=47.5)


@pytest.mark.parametrize(
    ("lat", "lng"),
    [(-90.1, 47.5), (90.1, 47.5), (-18.9, -180.1), (-18.9, 180.1)],
)
def test_borne_create_rejects_out_of_range_coordinates(lat: float, lng: float) -> None:
    with pytest.raises(ValidationError):
        BorneGpsCreate(name="Borne", lat=lat, lng=lng)


def test_borne_update_requires_a_non_blank_name() -> None:
    assert BorneGpsUpdate(name="  Borne sud ").name == "Borne sud"
    with pytest.raises(ValidationError):
        BorneGpsUpdate(name=" ")
