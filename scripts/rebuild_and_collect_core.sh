#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
echo "== pip info ==" > logs/pip_version.log
source venv/bin/activate
pip --version > logs/pip_version.log 2>&1 || true

echo "== uninstall sounddevice ==" > logs/pip_uninstall.log
pip uninstall -y sounddevice > logs/pip_uninstall.log 2>&1 || true

echo "== install sounddevice (build from source) ==" > logs/pip_install.log
pip install --no-binary :all: sounddevice > logs/pip_install.log 2>&1 || true


python - <<'PY' > logs/sd_info.log 2>&1 || true
import sys
try:
    import sounddevice as sd
    print('sd.__version__', getattr(sd, '__version__', 'unknown'))
    try:
        print('PortAudio version:', sd.get_portaudio_version())
    except Exception as e:
        print('get_portaudio_version error:', e)
    try:
        print('default.device', sd.default.device)
    except Exception as e:
        print('default.device error:', e)
except Exception as e:
    print('import sounddevice failed:', e)
PY


echo "== stress test (this may take a while) ==" > logs/stress.log
# keep going on errors; we want to capture crashes
ulimit -c unlimited
export MALLOC_CHECK_=3
export PYTHONFAULTHANDLER=1
python - <<'PY' >> logs/stress.log 2>&1 || true
import sounddevice as sd, numpy as np, sys
rates = [44100, 48000, 22050, 16000, 32000]
print('Starting stress loop', file=sys.stderr)
for i in range(2000):
    rate = rates[i % len(rates)]
    try:
        stream = sd.OutputStream(samplerate=rate, channels=1)
        stream.start()
        buf = np.zeros((int(rate*0.01),1), dtype='float32')
        stream.write(buf)
        stream.stop()
        stream.close()
    except Exception as e:
        print('Iteration', i, 'exception:', repr(e), file=sys.stderr)
    if i % 50 == 0:
        print('iter', i, file=sys.stderr)
    sys.stderr.flush()
print('Stress loop finished', file=sys.stderr)
PY

# locate core dump(s) and run gdb backtrace if found
corefile=$(ls -t core* 2>/dev/null | head -n1 || true)
echo "corefile=$corefile" > logs/core_check.log
if [ -n "$corefile" ]; then
  gdb --batch -ex "set pagination 0" -ex "thread apply all bt full" -ex "info threads" "$(which python)" "$corefile" > logs/gdb_bt.txt 2>&1 || true
  echo "GDB backtrace written to logs/gdb_bt.txt"
else
  echo "No core file found" >> logs/core_check.log
fi

echo "Done. See logs/ directory for outputs." 
