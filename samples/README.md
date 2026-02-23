This folder contains the following:

- **agents:**  sample code of various agents using Ephapsys SDK (TrustedAgent class)
- **modulators:**  sample code for modulating various models using Ephapsys SDK (ModulatorClient class)

## Install profiles for samples

Use the matching install command before running each sample:

| Sample type | Recommended install |
|---|---|
| `agents/helloworld` | `pip install "ephapsys[modulation]"` |
| `agents/robot` | `pip install "ephapsys[modulation,audio,vision,embedding]"` + `pip install webrtcvad sounddevice pyaudio` |
| `modulators/*` (training/modulation only) | `pip install "ephapsys[modulation]"` |
| `modulators/*` (full eval/report stack) | `pip install "ephapsys[all]"` |
