"""Nox sessions."""

import nox
from nox.sessions import Session


nox.options.sessions = "black", "flake8", "mypy", "tests"


@nox.session(python=False)
def black(session: Session):
    session.run("poetry", "install")
    session.run(
        "poetry", "run", "black",
        "src/rsschool_mlintro2022q1_capstone_project",
        "--line-length=79"
    )


@nox.session(python=False)
def flake8(session: Session):
    session.run(
        "poetry", "run", "flake8",
        "src/rsschool_mlintro2022q1_capstone_project"
    )


@nox.session(python=False)
def mypy(session: Session):
    session.run(
        "poetry", "run", "mypy",
        "src/rsschool_mlintro2022q1_capstone_project",
        "--config-file=mypy.ini"
    )


@nox.session(python=False)
def tests(session: Session):
    session.run(
        "poetry", "run", "test"
    )
