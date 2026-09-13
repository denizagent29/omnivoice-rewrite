#!/usr/bin/env python3
"""Окно для omnivoice-rewrite: всё то же самое, но мышкой и с горячими клавишами.

    python gui.py

Сделано под скринридер (NVDA на Windows):

* метка (StaticText) создаётся ДО своего поля — NVDA связывает их по порядку
  создания и озвучивает имя поля, иначе читается просто «Редактор»;
* списки выбора — wx.Choice: читается как список с текущим значением;
* результат и журнал — многострочные read-only поля: по ним удобно ходить
  стрелками и перечитывать;
* нативное меню с мнемониками (Alt), у всех действий горячие клавиши;
* ничего не блокирует окно: расшифровка и синтез идут в отдельном потоке, а
  прогресс и результат прилетают в строку состояния и проговариваются;
* если стоит accessible_output3, ключевые события озвучиваются отдельно.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

import wx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewrite  # noqa: E402

WILD_AUDIO = ("Звук (*.wav;*.mp3;*.ogg;*.m4a;*.flac)|*.wav;*.mp3;*.ogg;*.m4a;*.flac"
              "|Все файлы (*.*)|*.*")
WILD_ANY = "Все файлы (*.*)|*.*"
WILD_PT = "Голос (*.pt)|*.pt|" + WILD_ANY

DEVICES = ["cuda", "cpu", "mps", "xpu"]
ASR_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
PROVIDERS = ["deepseek", "moonshot", "gemini", "openai", "ollama", "none"]
TAGS = ["[laughter]", "[sigh]", "[question-en]", "[surprise-ah]", "[dissatisfaction-hnn]"]

_SPEECH = None


def speech(text: str):
    """Проговорить событие, если есть accessible_output3. Молча — если нет."""
    global _SPEECH
    try:
        if _SPEECH is None:
            from accessible_output3.outputs.auto import Auto

            _SPEECH = Auto()
        _SPEECH.output(text)
    except Exception:
        pass


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="OmniVoice — перепиши голосовое своим голосом",
                         size=(920, 720))
        self.cfg = wx.Config("omnivoice-rewrite")
        self.workdir = Path(self.cfg.Read("workdir", str(Path.home() / "omnivoice")))

        self._build_menu()
        self.book = wx.Notebook(self)
        self.book.AddPage(PipelinePanel(self.book, self), "Переписка")
        self.book.AddPage(SynthPanel(self.book, self), "Синтез")
        self.book.AddPage(AsrPanel(self.book, self), "Расшифровка")
        self.book.AddPage(SettingsPanel(self.book, self), "Настройки")
        self.book.AddPage(HelpPanel(self.book, self), "Справка")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.book, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.CreateStatusBar(1)
        self.set_status("Готов.")
        self.book.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_page_changed)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Centre()
        self.book.SetFocus()

    # ------------------------------------------------------------ меню

    def _build_menu(self):
        file_menu = wx.Menu()
        self.id_audio = wx.NewIdRef()
        file_menu.Append(self.id_audio, "Открыть &запись…\tCtrl+O",
                         "Выбрать голосовое сообщение")
        self.id_out = wx.NewIdRef()
        file_menu.Append(self.id_out, "Куда &сохранить…\tCtrl+Shift+S",
                         "Выбрать файл результата")
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "В&ыход\tCtrl+Q", "Закрыть окно")

        act_menu = wx.Menu()
        self.id_run = wx.NewIdRef()
        act_menu.Append(self.id_run, "&Всё сразу\tF5",
                        "Расшифровать, переписать и озвучить")
        self.id_transcribe = wx.NewIdRef()
        act_menu.Append(self.id_transcribe, "&Расшифровать\tCtrl+R", "Только расшифровка")
        self.id_edit = wx.NewIdRef()
        act_menu.Append(self.id_edit, "&Переписать\tCtrl+E", "Только правка текста у LLM")
        self.id_synth = wx.NewIdRef()
        act_menu.Append(self.id_synth, "&Озвучить\tCtrl+S", "Только синтез голосом")

        help_menu = wx.Menu()
        self.id_help = wx.NewIdRef()
        help_menu.Append(self.id_help, "&Клавиши\tF1", "Список горячих клавиш")

        bar = wx.MenuBar()
        bar.Append(file_menu, "&Файл")
        bar.Append(act_menu, "&Действия")
        bar.Append(help_menu, "&Справка")
        self.SetMenuBar(bar)
        self.SetAcceleratorTable(wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord("O"), self.id_audio),
            (wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord("S"), self.id_out),
            (wx.ACCEL_CTRL, ord("Q"), wx.ID_EXIT),
            (wx.ACCEL_NORMAL, wx.WXK_F5, self.id_run),
            (wx.ACCEL_CTRL, ord("R"), self.id_transcribe),
            (wx.ACCEL_CTRL, ord("E"), self.id_edit),
            (wx.ACCEL_CTRL, ord("S"), self.id_synth),
            (wx.ACCEL_NORMAL, wx.WXK_F1, self.id_help),
        ]))
        self.Bind(wx.EVT_MENU, self.on_choose_audio, id=self.id_audio)
        self.Bind(wx.EVT_MENU, self.on_choose_out, id=self.id_out)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, lambda e: self.page("run"), id=self.id_run)
        self.Bind(wx.EVT_MENU, lambda e: self.page("transcribe"), id=self.id_transcribe)
        self.Bind(wx.EVT_MENU, lambda e: self.page("edit"), id=self.id_edit)
        self.Bind(wx.EVT_MENU, lambda e: self.page("synth"), id=self.id_synth)
        self.Bind(wx.EVT_MENU, self.on_help, id=self.id_help)

    def on_page_changed(self, event):
        # При смене вкладки уводим фокус внутрь неё: иначе NVDA остаётся на
        # списке вкладок и Tab уходит не туда.
        page = self.book.GetPage(event.GetSelection())
        if getattr(page, "first", None):
            page.first.SetFocus()
        event.Skip()

    def page(self, action: str):
        """Пункты меню работают с той вкладкой, на которой стоит фокус."""
        panel = self.book.GetCurrentPage()
        fn = getattr(panel, "menu_" + action, None)
        if fn:
            fn()
        else:
            self.set_status("Это действие не подходит текущей вкладке.")
            speech("Это действие не подходит текущей вкладке.")

    def on_help(self, _event):
        wx.MessageBox(
            "Ctrl+O — выбрать запись\nCtrl+R — расшифровать\nCtrl+E — переписать\n"
            "Ctrl+S — озвучить\nF5 — всё сразу\nCtrl+Shift+S — куда сохранить\n"
            "Ctrl+Q — выход\n\nВкладки переключаются Ctrl+Tab, поля — Tab.",
            "Горячие клавиши", wx.OK | wx.ICON_INFORMATION, self)

    def on_choose_audio(self, _event):
        panel = self.book.GetCurrentPage()
        if hasattr(panel, "choose_audio"):
            panel.choose_audio()

    def on_choose_out(self, _event):
        panel = self.book.GetCurrentPage()
        if hasattr(panel, "choose_out"):
            panel.choose_out()

    # ------------------------------------------------------------ общее

    def set_status(self, text: str):
        self.SetStatusText(text.replace("\n", " ")[:220])

    def make_workdir(self) -> Path:
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir

    def run_bg(self, title: str, fn, on_done=None):
        """Работа в фоне: окно не замирает, NVDA не теряет фокус."""
        self.set_status(title)
        speech(title)

        def body():
            try:
                result = fn()
                if on_done:
                    wx.CallAfter(on_done, result)
            except SystemExit as e:  # rewrite.die() бросает SystemExit
                wx.CallAfter(self.fail, str(e))
            except Exception:
                wx.CallAfter(self.fail, traceback.format_exc())

        threading.Thread(target=body, daemon=True).start()

    def fail(self, text: str):
        self.set_status("Ошибка.")
        speech("Ошибка.")
        wx.MessageBox(text, "Не получилось", wx.OK | wx.ICON_ERROR, self)

    def open_path(self, path: str):
        if path and Path(path).exists():
            wx.LaunchDefaultApplication(path)
        else:
            self.fail("Файл ещё не готов: " + (path or "путь не задан"))

    def on_close(self, _event):
        self.cfg.Write("workdir", str(self.workdir))
        self.cfg.Flush()
        self.Destroy()


class Panel(wx.Panel):
    """ Общая обвязка: метка до поля, поле в сайзере, фокус на первом поле."""

    def __init__(self, parent, frame: MainFrame):
        super().__init__(parent)
        self.frame = frame
        self.grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=8)
        self.grid.AddGrowableCol(1, 1)
        self.first = None

    def field(self, label: str, ctrl: wx.Window, grow: bool = True):
        # StaticText строго перед контролом — иначе NVDA не свяжет имя и поле.
        self.grid.Add(wx.StaticText(self, label=label), 0,
                      wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        self.grid.Add(ctrl, 1 if grow else 0, wx.EXPAND)
        if self.first is None:
            self.first = ctrl
        return ctrl

    def file_field(self, label: str, value: str, cb, wildcard: str = WILD_ANY) -> wx.TextCtrl:
        """Текстовое поле с кнопкой «Обзор…»: NVDA видит обычный редактор."""
        ctrl = wx.TextCtrl(self, value=value)
        btn = wx.Button(self, label="Обзор…", style=wx.BU_EXACTFIT)
        btn.Bind(wx.EVT_BUTTON, lambda ev: cb(ctrl, wildcard))
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(ctrl, 1, wx.EXPAND | wx.RIGHT, 6)
        row.Add(btn, 0)
        holder = wx.Panel(self)
        holder.SetSizer(row)
        self.field(label, holder)
        if self.first is holder:
            self.first = ctrl
        return ctrl

    def text_row(self, label: str, value: str = "") -> wx.TextCtrl:
        c = wx.TextCtrl(self, value=value,
                        style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        c.SetMinSize((-1, 90))
        c.SetFont(wx.Font(wx.FontInfo(10).Family(wx.FONTFAMILY_TELETYPE)))
        self.grid.Add(wx.StaticText(self, label=label), 0, wx.ALIGN_TOP | wx.ALIGN_RIGHT)
        self.grid.Add(c, 1, wx.EXPAND)
        return c

    def button_row(self, *buttons):
        row = wx.BoxSizer(wx.HORIZONTAL)
        for b in buttons:
            row.Add(b, 0, wx.RIGHT, 8)
            if self.first is None:
                self.first = b
        self.grid.Add((0, 0))
        self.grid.Add(row, 0)

    def finish(self):
        outer = wx.BoxSizer(wx.VERTICAL)
        outer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 12)
        self.SetSizer(outer)
        if self.first:
            self.first.SetFocus()


def pick_file(panel: wx.Window, ctrl: wx.TextCtrl, wildcard: str, title: str,
              save: bool = False):
    style = (wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) if save else (wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    with wx.FileDialog(panel, title, defaultFile=Path(ctrl.GetValue()).name or None,
                       wildcard=wildcard, style=style) as dlg:
        if dlg.ShowModal() == wx.ID_OK:
            ctrl.SetValue(dlg.GetPath())
            speech(Path(dlg.GetPath()).name)


# ---------------------------------------------------------------- переписка

class PipelinePanel(Panel):
    def __init__(self, parent, frame):
        super().__init__(parent, frame)
        cfg = frame.cfg
        self.audio = self.file_field("Запись", cfg.Read("audio", ""),
                                     lambda c, w: pick_file(self, c, WILD_AUDIO, "Выберите запись"))
        self.instruction = self.field(
            "Что сделать с текстом",
            wx.TextCtrl(self, value=cfg.Read("instruction", "перепиши как полную противоположность")))
        self.provider = self.field("Модель для правки", wx.Choice(self, choices=PROVIDERS))
        self.provider.SetStringSelection(cfg.Read("provider", os.environ.get("LLM_PROVIDER", "deepseek")))
        self.out = self.file_field(
            "Файл результата", cfg.Read("out", str(frame.workdir / "result.wav")),
            lambda c, w: pick_file(self, c, WILD_AUDIO, "Куда сохранить результат", save=True))

        self.transcript = self.text_row("Расшифровка")
        self.edited = self.text_row("После правки")
        names = ["&Расшифровать", "&Переписать", "&Озвучить", "Всё ср&азу", "Прои&грать",
                 "&Открыть папку"]
        self.buttons = tuple(wx.Button(self, label=n) for n in names)
        self.button_row(*self.buttons)
        t, e, s, a, play, folder = self.buttons
        t.Bind(wx.EVT_BUTTON, lambda ev: self.menu_transcribe())
        e.Bind(wx.EVT_BUTTON, lambda ev: self.menu_edit())
        s.Bind(wx.EVT_BUTTON, lambda ev: self.menu_synth())
        a.Bind(wx.EVT_BUTTON, lambda ev: self.menu_run())
        play.Bind(wx.EVT_BUTTON, lambda ev: self.frame.open_path(self.out.GetValue()))
        folder.Bind(wx.EVT_BUTTON, lambda ev: self.frame.open_path(
            str(Path(self.out.GetValue()).parent)))
        self.finish()

    def choose_audio(self):
        pick_file(self, self.audio, WILD_AUDIO, "Выберите запись")

    def choose_out(self):
        pick_file(self, self.out, WILD_AUDIO, "Куда сохранить результат", save=True)

    def _values(self):
        return (self.audio.GetValue(), self.instruction.GetValue(),
                self.provider.GetStringSelection(), self.out.GetValue())

    def menu_transcribe(self):
        audio, _, _, _ = self._values()
        if not audio:
            return self.frame.fail("Сначала выберите запись.")
        model = self.frame.cfg.Read("asr_model", "small")
        lang = self.frame.cfg.Read("asr_lang", "") or None

        def done(res):
            self.transcript.SetValue(res["text"])
            self.edited.SetValue("")
            self.frame.set_status(f"Расшифровано, язык {res['language']}.")
            speech("Расшифровка готова.")

        self.frame.run_bg("Расшифровываю…",
                          lambda: rewrite.transcribe(audio, model, lang, "auto"), done)

    def menu_edit(self):
        text = self.transcript.GetValue().strip()
        if not text:
            return self.frame.fail("Сначала расшифруйте запись.")
        _, instruction, provider, _ = self._values()

        def done(new_text):
            self.edited.SetValue(new_text)
            self.frame.set_status("Текст переписан.")
            speech("Текст переписан.")

        self.frame.run_bg("Переписываю текст…",
                          lambda: rewrite.edit(text, instruction, provider), done)

    def _synth_job(self, text, audio, out):
        """Синтез из вкладки «Переписка»: образец голоса — из той же записи."""
        import tempfile

        device = self.frame.cfg.Read("device", "cuda")
        num_step = self.frame.cfg.ReadInt("num_step", 32)

        def job():
            tr = rewrite.transcribe(audio, "small", None, "cpu")
            with tempfile.TemporaryDirectory() as td:
                ref, ref_text = rewrite.pick_ref(audio, tr["segments"], Path(td))
                rewrite.say_omnivoice(text, Path(out), ref=ref, ref_text=ref_text,
                                      device=device, num_step=num_step)
        return job

    def _synth_done(self, out):
        def done(_):
            self.frame.set_status("Готово: " + out)
            speech("Готово. Файл сохранён.")
        return done

    def menu_synth(self):
        text = self.edited.GetValue().strip() or self.transcript.GetValue().strip()
        if not text:
            return self.frame.fail("Нет текста для синтеза.")
        audio, _, _, out = self._values()
        if not audio:
            return self.frame.fail("Нужна запись-образец голоса.")
        self.frame.run_bg("Синтезирую голосом…", self._synth_job(text, audio, out),
                          self._synth_done(out))

    def menu_run(self):
        audio, instruction, provider, out = self._values()
        if not audio:
            return self.frame.fail("Сначала выберите запись.")
        model = self.frame.cfg.Read("asr_model", "small")

        def job():
            tr = rewrite.transcribe(audio, model, None, "auto")
            wx.CallAfter(self.transcript.SetValue, tr["text"])
            new_text = rewrite.edit(tr["text"], instruction, provider)
            wx.CallAfter(self.edited.SetValue, new_text)
            # Синтез внутри того же потока: модель уже загрузится один раз.
            import tempfile

            device = self.frame.cfg.Read("device", "cuda")
            num_step = self.frame.cfg.ReadInt("num_step", 32)
            with tempfile.TemporaryDirectory() as td:
                ref, ref_text = rewrite.pick_ref(audio, tr["segments"], Path(td))
                rewrite.say_omnivoice(new_text, Path(out), ref=ref, ref_text=ref_text,
                                      device=device, num_step=num_step)

        self.frame.run_bg("Делаю всё: расшифровка, правка, синтез…", job, self._synth_done(out))


# ---------------------------------------------------------------- синтез

class SynthPanel(Panel):
    """Полный OmniVoice: клонирование, дизайн голоса, авто-голос и все параметры."""

    def __init__(self, parent, frame):
        super().__init__(parent, frame)
        cfg = frame.cfg
        self.mode = self.field("Режим", wx.Choice(self, choices=[
            "Клонирование по образцу", "Дизайн по описанию", "Авто-голос"]))
        self.mode.SetSelection(cfg.ReadInt("mode", 0))

        self.text = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.text.SetMinSize((-1, 130))
        self.field("Текст", self.text)

        self.ref = self.file_field("Образец голоса", cfg.Read("synth_ref", ""),
                                   lambda c, w: pick_file(self, c, WILD_AUDIO, "Образец голоса"))
        self.ref_text = self.field("Расшифровка образца (пусто — распознает сам)",
                                   wx.TextCtrl(self))
        self.instruct = self.field("Описание голоса (для дизайна)",
                                   wx.TextCtrl(self, value=cfg.Read("instruct", "female, low pitch")))
        self.voice = self.file_field("Готовый голос (.pt)", cfg.Read("voice", ""),
                                     lambda c, w: pick_file(self, c, WILD_PT, "Готовый голос"))

        self.num_step = self.field("Шаги диффузии", wx.SpinCtrl(
            self, min=4, max=128, initial=cfg.ReadInt("num_step", 32)))
        self.guidance = self.field("Следование тексту", wx.SpinCtrlDouble(
            self, min=0.0, max=10.0, inc=0.1, initial=cfg.ReadDouble("guidance", 2.0)))
        self.speed = self.field("Темп (0 — как получится)", wx.SpinCtrlDouble(
            self, min=0.0, max=3.0, inc=0.05, initial=cfg.ReadDouble("speed", 0.0)))
        self.out = self.file_field("Файл результата", cfg.Read("synth_out", str(frame.workdir / "synth.wav")),
                                   lambda c, w: pick_file(self, c, WILD_AUDIO, "Куда сохранить синтез", save=True))

        names = ["&Синтезировать", "Сохранить &голос", "Прои&грать", "Вставить &звук"]
        self.buttons = tuple(wx.Button(self, label=n) for n in names)
        self.button_row(*self.buttons)
        go, save, play, tag = self.buttons
        go.Bind(wx.EVT_BUTTON, lambda ev: self.menu_synth())
        save.Bind(wx.EVT_BUTTON, lambda ev: self.save_voice())
        play.Bind(wx.EVT_BUTTON, lambda ev: self.frame.open_path(self.out.GetValue()))
        tag.Bind(wx.EVT_BUTTON, lambda ev: self.insert_tag())
        self.finish()

    def choose_audio(self):
        pick_file(self, self.ref, WILD_AUDIO, "Образец голоса")

    def choose_out(self):
        pick_file(self, self.out, WILD_AUDIO, "Куда сохранить синтез", save=True)

    def insert_tag(self):
        with wx.SingleChoiceDialog(self, "Какой звук вставить?", "Неречевой звук", TAGS) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.text.SetValue(self.text.GetValue() + " " + dlg.GetStringSelection())

    def save_voice(self):
        ref = self.ref.GetValue().strip()
        if not ref:
            return self.frame.fail("Для сохранения голоса нужен образец.")
        with wx.FileDialog(self, "Куда сохранить голос", defaultFile="my_voice.pt",
                           wildcard=WILD_PT, style=wx.FD_SAVE) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            target = dlg.GetPath()
        device = self.frame.cfg.Read("device", "cuda")
        ref_text = self.ref_text.GetValue().strip()

        def job():
            model = rewrite.load_model(device)
            prompt = model.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text or None)
            prompt.save(target)
            return target

        def done(path):
            self.voice.SetValue(str(path))
            self.frame.set_status("Голос сохранён: " + str(path))
            speech("Голос сохранён.")

        self.frame.run_bg("Сохраняю голос…", job, done)

    def menu_synth(self):
        text = self.text.GetValue().strip()
        if not text:
            return self.frame.fail("Введите текст.")
        mode = self.mode.GetSelection()
        device = self.frame.cfg.Read("device", "cuda")
        out = self.out.GetValue()
        prompt_path = self.voice.GetValue().strip()
        ref = self.ref.GetValue().strip()
        ref_text = self.ref_text.GetValue().strip()
        instruct = self.instruct.GetValue().strip()
        num_step = self.num_step.GetValue()
        guidance = float(self.guidance.GetValue())
        speed = float(self.speed.GetValue()) or None

        def job():
            common = dict(device=device, num_step=num_step, guidance_scale=guidance, speed=speed)
            if prompt_path:
                rewrite.say_omnivoice(text, Path(out), prompt_path=prompt_path, **common)
            elif mode == 0 and ref:
                rewrite.say_omnivoice(text, Path(out), ref=Path(ref), ref_text=ref_text, **common)
            elif mode == 1 and instruct:
                rewrite.say_omnivoice(text, Path(out), instruct=instruct, **common)
            else:
                rewrite.say_omnivoice(text, Path(out), **common)

        def done(_):
            self.frame.set_status("Готово: " + out)
            speech("Синтез готов.")

        self.frame.run_bg("Синтезирую…", job, done)


# ---------------------------------------------------------------- расшифровка

class AsrPanel(Panel):
    def __init__(self, parent, frame):
        super().__init__(parent, frame)
        cfg = frame.cfg
        self.audio = self.file_field("Запись", cfg.Read("audio", ""),
                                     lambda c, w: pick_file(self, c, WILD_AUDIO, "Выберите запись"))
        self.model = self.field("Размер модели", wx.Choice(self, choices=ASR_MODELS))
        self.model.SetStringSelection(cfg.Read("asr_model", "small"))
        self.lang = self.field("Язык (пусто — определит сам)",
                               wx.TextCtrl(self, value=cfg.Read("asr_lang", "")))
        self.device = self.field("Где считать", wx.Choice(self, choices=["auto"] + DEVICES))
        self.device.SetStringSelection(cfg.Read("asr_device", "auto"))
        self.result = self.text_row("Расшифровка")
        names = ["Рас&шифровать", "Сох&ранить в файл"]
        self.buttons = tuple(wx.Button(self, label=n) for n in names)
        self.button_row(*self.buttons)
        self.buttons[0].Bind(wx.EVT_BUTTON, lambda ev: self.menu_transcribe())
        self.buttons[1].Bind(wx.EVT_BUTTON, lambda ev: self.save_text())
        self.finish()

    def choose_audio(self):
        pick_file(self, self.audio, WILD_AUDIO, "Выберите запись")

    def save_text(self):
        text = self.result.GetValue()
        if not text:
            return self.frame.fail("Нечего сохранять.")
        with wx.FileDialog(self, "Сохранить расшифровку", defaultFile="transcript.txt",
                           wildcard="Текст (*.txt)|*.txt", style=wx.FD_SAVE) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                Path(dlg.GetPath()).write_text(text, encoding="utf-8")
                self.frame.set_status("Расшифровка сохранена.")
                speech("Расшифровка сохранена.")

    def menu_transcribe(self):
        audio = self.audio.GetValue().strip()
        if not audio:
            return self.frame.fail("Выберите запись.")
        model = self.model.GetStringSelection()
        lang = self.lang.GetValue().strip() or None
        device = self.device.GetStringSelection()
        self.frame.cfg.Write("asr_model", model)
        self.frame.cfg.Write("asr_lang", self.lang.GetValue().strip())
        self.frame.cfg.Write("asr_device", device)

        def done(res):
            self.result.SetValue(res["text"])
            self.frame.set_status(f"Язык: {res['language']}. Готово.")
            speech("Расшифровка готова.")

        self.frame.run_bg("Расшифровываю…",
                          lambda: rewrite.transcribe(audio, model, lang, device), done)


# ---------------------------------------------------------------- настройки

class SettingsPanel(Panel):
    def __init__(self, parent, frame):
        super().__init__(parent, frame)
        cfg = frame.cfg
        self.provider = self.field("Провайдер LLM", wx.Choice(self, choices=PROVIDERS))
        self.provider.SetStringSelection(cfg.Read("provider", os.environ.get("LLM_PROVIDER", "deepseek")))
        self.key = self.field("Ключ LLM (на диск не сохраняется)",
                              wx.TextCtrl(self, style=wx.TE_PASSWORD))
        self.base = self.field("Свой адрес LLM (необязательно)",
                               wx.TextCtrl(self, value=cfg.Read("llm_base", "")))
        self.llm_model = self.field("Модель LLM (необязательно)",
                                    wx.TextCtrl(self, value=cfg.Read("llm_model", "")))
        self.device = self.field("Устройство для синтеза", wx.Choice(self, choices=DEVICES))
        self.device.SetStringSelection(cfg.Read("device", "cuda"))
        self.workdir = self.file_field("Папка результатов", str(frame.workdir),
                                       self._choose_dir)
        names = ["&Сохранить", "Про&верить ключ"]
        self.buttons = tuple(wx.Button(self, label=n) for n in names)
        self.button_row(*self.buttons)
        self.buttons[0].Bind(wx.EVT_BUTTON, lambda ev: self.save())
        self.buttons[1].Bind(wx.EVT_BUTTON, lambda ev: self.check_key())
        self.finish()

    def _choose_dir(self, ctrl, _wildcard):
        with wx.DirDialog(self, "Папка результатов") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                ctrl.SetValue(dlg.GetPath())

    def choose_audio(self):
        self.workdir.SetValue(str(self.frame.workdir))

    def save(self):
        cfg = self.frame.cfg
        cfg.Write("provider", self.provider.GetStringSelection())
        cfg.Write("llm_base", self.base.GetValue())
        cfg.Write("llm_model", self.llm_model.GetValue())
        cfg.Write("device", self.device.GetStringSelection())
        if self.workdir.GetValue().strip():
            self.frame.workdir = Path(self.workdir.GetValue().strip())
            self.frame.make_workdir()
        os.environ["LLM_PROVIDER"] = self.provider.GetStringSelection()
        if self.key.GetValue().strip():
            # Ключ в реестр не пишем: он живёт до закрытия окна.
            os.environ["LLM_API_KEY"] = self.key.GetValue().strip()
        if self.base.GetValue().strip():
            os.environ["LLM_BASE_URL"] = self.base.GetValue().strip()
        if self.llm_model.GetValue().strip():
            os.environ["LLM_MODEL"] = self.llm_model.GetValue().strip()
        cfg.Flush()
        self.frame.set_status("Настройки сохранены.")
        speech("Настройки сохранены.")

    def check_key(self):
        self.save()
        provider = self.provider.GetStringSelection()

        def done(text):
            wx.MessageBox("Ответ модели:\n\n" + text, "Ключ работает",
                          wx.OK | wx.ICON_INFORMATION, self)
            self.frame.set_status("Ключ работает.")
            speech("Ключ работает.")

        self.frame.run_bg("Проверяю ключ…",
                          lambda: rewrite.edit("Это проверка связи.", "Ответь одним словом: работает.",
                                               provider), done)


# ---------------------------------------------------------------- справка

HELP = """OmniVoice — три режима синтеза

Клонирование по образцу. Даёте запись 3–10 секунд и её расшифровку — модель
говорит вашим голосом. Кнопка «Сохранить голос» кладёт тембр в файл .pt, и
дальше образец больше не нужен.

Дизайн по описанию. Образец не нужен: голос описывается словами — пол, возраст,
высота, акцент, шёпот. Устойчиво работает на английском и китайском, на других
языках возможны сюрпризы.

Авто-голос. Ни образца, ни описания — модель выбирает голос сама.

Параметры

Шаги диффузии: 16 быстрее, 32 чище, дальше смысла мало.
Следование тексту: насколько строго держаться написанного.
Темп: больше единицы — быстрее, меньше — медленнее; ноль — как получится.

Неречевые звуки

В текст можно вставлять теги: [laughter], [sigh], [question-en], [surprise-ah]
и другие — модель произнесёт их звуком, а не словом. Кнопка «Вставить звук»
подставляет выбранный тег в конец текста.

Длинный текст

Модель сама режет длинный текст на куски и держит память почти постоянной:
можно отдавать хоть целую главу.

Горячие клавиши

Ctrl+O выбрать запись, Ctrl+R расшифровать, Ctrl+E переписать, Ctrl+S озвучить,
F5 всё сразу, Ctrl+Shift+S куда сохранить, Ctrl+Tab сменить вкладку, F1 справка.

Ключ LLM

Вводится во вкладке «Настройки», на диск не сохраняется и живёт до закрытия
окна. Переменные LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL работают
точно так же — ими удобнее пользоваться из командной строки.
"""


class HelpPanel(Panel):
    def __init__(self, parent, frame):
        super().__init__(parent, frame)
        self.grid.Add((0, 0))
        self.view = self.text_row("", HELP)
        self.view.SetMinSize((-1, 430))
        self.finish()


def main():
    app = wx.App(False)
    frame = MainFrame()
    frame.Show(True)
    app.MainLoop()


if __name__ == "__main__":
    main()
