"""Modal dialog for editing the custom final-report prompt.

Minimal on purpose: one explanatory line, a ``QPlainTextEdit`` and
OK/Cancel. The caller passes the text to start from (the stored custom
text, or the built-in default prompt as a starting template when the
stored text is empty) and gets back the edited text, or ``None`` when
the user cancelled.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


def edit_report_prompt(parent: QWidget, text: str) -> Optional[str]:
    """Show the modal prompt editor; return the edited text, or None on cancel."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Промпт итогового отчёта")
    dialog.resize(560, 420)

    layout = QVBoxLayout(dialog)
    layout.addWidget(
        QLabel(
            "Опишите, каким должен быть итоговый отчёт: разделы, порядок, "
            "степень детализации. Стенограмма или промежуточные конспекты "
            "подставляются автоматически."
        )
    )
    edit = QPlainTextEdit(dialog)
    edit.setPlainText(text)
    layout.addWidget(edit, stretch=1)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    # Read the result BEFORE deleteLater: the deletion is only scheduled
    # here and processed by the event loop after we return, so `edit` is
    # still alive at this point.
    text = (
        edit.toPlainText()
        if dialog.exec() == QDialog.DialogCode.Accepted
        else None
    )
    # The dialog has no parent-ownership cleanup of its own: without this,
    # every open/close would leak one child of `parent`.
    dialog.deleteLater()
    return text
