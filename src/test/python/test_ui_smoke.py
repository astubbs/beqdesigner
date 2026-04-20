"""Smoke tests for the Qt UI layer.

These exist to prove the pytest-qt harness is wired up so future contributors
can add real dialog tests without fighting plumbing. Each test instantiates a
generated Qt Designer form, shows the widget, and immediately closes it.
"""

from qtpy.QtWidgets import QDialog

from ui.values import Ui_valuesDialog


class _ValuesHost(QDialog, Ui_valuesDialog):
    """Minimal host providing the slot the generated Ui expects."""

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.last_value: float | None = None

    def updateValues(self, value: float) -> None:
        self.last_value = value


def test_values_dialog_opens(qtbot):
    dialog = _ValuesHost()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.isVisible()
    assert dialog.windowTitle() == "Values"


def test_values_dialog_signal_connects(qtbot):
    dialog = _ValuesHost()
    qtbot.addWidget(dialog)
    # The generated Ui wires freq.valueChanged to updateValues -- changing the
    # spin box should drive the host's state via the connected slot.
    dialog.freq.setValue(123.4)
    assert dialog.last_value == 123.4
