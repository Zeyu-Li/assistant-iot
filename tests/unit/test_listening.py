from __future__ import annotations

from assistant.core.listening import ListeningController, ListeningMode


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_wake_word_mode_needs_wake_word_or_follow_up() -> None:
    clock = FakeClock()
    controller = ListeningController(ListeningMode.WAKE_WORD, follow_up_seconds=8, clock=clock)

    assert not controller.accepts_speech(wake_word_detected=False)
    assert controller.accepts_speech(wake_word_detected=True)

    controller.open_follow_up()
    clock.now += 7.9
    assert controller.accepts_speech(wake_word_detected=False)
    clock.now += 0.2
    assert not controller.accepts_speech(wake_word_detected=False)


def test_muted_ignores_everything() -> None:
    controller = ListeningController(ListeningMode.MUTED, follow_up_seconds=8)
    assert not controller.accepts_speech(wake_word_detected=True)


def test_open_mic_accepts_all_but_checks_addressee_without_wake_word() -> None:
    clock = FakeClock()
    controller = ListeningController(ListeningMode.OPEN_MIC, follow_up_seconds=8, clock=clock)

    assert controller.accepts_speech(wake_word_detected=False)
    assert controller.needs_addressed_check(wake_word_detected=False)
    assert not controller.needs_addressed_check(wake_word_detected=True)
    controller.open_follow_up()
    assert not controller.needs_addressed_check(wake_word_detected=False)


def test_mode_change_notifies_and_closes_follow_up() -> None:
    controller = ListeningController(ListeningMode.WAKE_WORD, follow_up_seconds=8)
    changes: list[ListeningMode] = []
    controller.on_change(changes.append)
    controller.open_follow_up()

    controller.set_mode(ListeningMode.OPEN_MIC)
    controller.set_mode(ListeningMode.OPEN_MIC)

    assert changes == [ListeningMode.OPEN_MIC]
    assert not controller.in_follow_up()
