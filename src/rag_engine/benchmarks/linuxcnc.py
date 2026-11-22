"""Original evaluation cases for the LinuxCNC 2.9 documentation."""

LINUXCNC_DOCUMENTS = {
    "config/stepper-diagnostics.html": "https://linuxcnc.org/docs/2.9/html/config/stepper-diagnostics.html",
    "install/latency-test.html": "https://linuxcnc.org/docs/2.9/html/install/latency-test.html",
}


def _case(case_id, question, answer=None, *sources, category="diagnostics"):
    return {
        "id": f"linuxcnc-{case_id}",
        "question": question,
        "answer": answer,
        "answerable": answer is not None,
        "category": category,
        "references": [
            {
                "source": (
                    "config/stepper-diagnostics.html"
                    if source.startswith("stepper-diagnostics/")
                    else "install/latency-test.html"
                ),
                "section": source.removesuffix(".md").split("/")[-1],
            }
            for source in sources
        ],
    }


LINUXCNC_CASES = [
    _case(
        "01",
        "After a fresh LinuxCNC 2.9 installation, alternating jog directions moves "
        "my stepper one pulse each time, always in the same direction. What signal "
        "wiring mistake should I look for first?",
        "Check whether the step and direction signals were swapped.",
        "stepper-diagnostics/stepper_moves_one_step.md",
    ),
    _case(
        "02",
        "On my LinuxCNC 2.9 machine none of the steppers move. What common "
        "drive-level prerequisite should I verify?",
        "Check whether the drives need an asserted enable input or a charge pump.",
        "stepper-diagnostics/no_steppers_move.md",
    ),
    _case(
        "03",
        "The axis moves, but the measured displacement differs from the commanded "
        "distance by a consistent factor. Which LinuxCNC 2.9 setting should I inspect?",
        "Inspect the axis scale setting.",
        "stepper-diagnostics/distance_not_correct.md",
    ),
    _case(
        "04",
        "LinuxCNC 2.9 shows a following error on an open-loop stepper. How can it "
        "detect an error without encoder feedback?",
        "It determines whether it can keep up with the commanded motion. The error "
        "does not require feedback about the motor's actual position.",
        "stepper-diagnostics/following_error.md",
    ),
    _case(
        "05",
        "I enabled backlash compensation and then saw following errors. Which "
        "acceleration setting does the LinuxCNC 2.9 guide say to increase on the "
        "affected axes, and by how much?",
        "Increase STEPGEN_MAXACCEL on each affected axis, up to twice that axis's "
        "MAX_ACCELERATION, to provide extra acceleration at reversals.",
        "stepper-diagnostics/following_error.md",
    ),
    _case(
        "06",
        "LinuxCNC 2.9 showed an unexpected real-time delay once during a session. "
        "Which test should I run to investigate intermittent delays, and should "
        "LinuxCNC or StepConf be running at the same time?",
        "Run the latency test for an extended period. Close LinuxCNC and StepConf "
        "while the test is running.",
        "stepper-diagnostics/rtapi_error.md",
        "latency-test/latency_tests.md",
    ),
    _case(
        "07",
        "On LinuxCNC 2.9, which measurement from latency-test matters most when "
        "evaluating a computer for software stepping?",
        "Use the maximum jitter of the base thread.",
        "latency-test/latency_test.md",
    ),
    _case(
        "08",
        "My LinuxCNC 2.9 system logged one real-time delay at startup. Which exact "
        "peripheral caused it?",
        category="missing machine context",
    ),
    _case(
        "09",
        "The first jog after installing LinuxCNC 2.9 did not move an axis. What "
        "voltage should I apply to the drive's enable pin? I don't know the drive model.",
        category="missing drive model",
    ),
    _case(
        "10",
        "Do the LinuxCNC 2.9 stepper diagnostics confirm the default latency test "
        "periods for LinuxCNC 2.10?",
        category="version mismatch",
    ),
]
