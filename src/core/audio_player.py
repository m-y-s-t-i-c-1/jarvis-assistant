"""
Player audio persistent: un singur `OutputStream` deschis și un thread
care extrage buffere dintr-o coadă FIFO. Evită deschideri/închideri
repetate ale PortAudio care pot cauza corupții native.

API simplu:
 - play_blocking(audio: np.ndarray, sample_rate: int, stop_event: threading.Event)
 - stop_current()

"""
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from src.core.audio_io import DEVICE_REDARE_DIFUZOARE, _OPEN_CLOSE_LOCK


class _PlayerThread:
    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioPlayer")
        self._shutdown = threading.Event()
        self._current_stop = None
        self._stream = None
        self._thread.start()


    def _close_stream(self):
        # No persistent OutputStream in this implementation; keep as noop
        self._stream = None

    def _run(self):
        while not self._shutdown.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            audio = item["audio"]
            sr = item["sr"]
            done_ev = item["done_ev"]
            stop_ev = item["stop_ev"]

            # Determine channels
            channels = 1 if audio.ndim == 1 else audio.shape[1]

            # Query device sample rate
            try:
                info = sd.query_devices(DEVICE_REDARE_DIFUZOARE)
                dev_sr = int(info.get("default_samplerate", sr))
            except Exception:
                dev_sr = sr

            # Resample if needed
            if dev_sr != sr:
                try:
                    audio = resample_poly(audio, dev_sr, sr, axis=0).astype(np.float32)
                except Exception:
                    # fallback: simple interp
                    n_new = max(1, int(audio.shape[0] * dev_sr / sr))
                    idx_new = np.linspace(0, audio.shape[0] - 1, n_new)
                    if audio.ndim == 1:
                        audio = np.interp(idx_new, np.arange(audio.shape[0]), audio).astype(np.float32)
                    else:
                        channels_list = [
                            np.interp(idx_new, np.arange(audio.shape[0]), audio[:, c])
                            for c in range(audio.shape[1])
                        ]
                        audio = np.stack(channels_list, axis=1).astype(np.float32)

            # Use sd.play in the worker thread to avoid repeated OutputStream
            # open/close races observed on some systems.
            self._current_stop = stop_ev
            try:
                try:
                    with _OPEN_CLOSE_LOCK:
                        sd.play(audio, samplerate=dev_sr, device=DEVICE_REDARE_DIFUZOARE)

                    # Wait in small steps so we can react to stop_ev
                    while sd.get_stream() is not None and not stop_ev.is_set():
                        time.sleep(0.01)

                    if stop_ev.is_set():
                        try:
                            with _OPEN_CLOSE_LOCK:
                                sd.stop()
                        except Exception:
                            pass
                except Exception:
                    # If sd.play fails, mark done and continue
                    done_ev.set()
                    self._current_stop = None
                    continue
            finally:
                done_ev.set()
                self._current_stop = None

        # shutdown: close stream
        self._close_stream()

    def play_blocking(self, audio: np.ndarray, sr: int, stop_event: threading.Event):
        done = threading.Event()
        stop_local = threading.Event()
        # Compose queue item
        item = {"audio": audio, "sr": sr, "done_ev": done, "stop_ev": stop_local}
        self._q.put(item)

        # Wait until done or external stop
        while not done.is_set():
            if stop_event.is_set():
                stop_local.set()
            time.sleep(0.01)

    def stop_current(self):
        if self._current_stop is not None:
            try:
                self._current_stop.set()
            except Exception:
                pass

    def shutdown(self):
        self._shutdown.set()
        self._q.put(None)
        self._thread.join(timeout=1.0)


# Singleton
_player = _PlayerThread()


def play_blocking(audio: np.ndarray, sr: int, stop_event: threading.Event):
    return _player.play_blocking(audio, sr, stop_event)


def stop_current():
    return _player.stop_current()


def shutdown():
    return _player.shutdown()
