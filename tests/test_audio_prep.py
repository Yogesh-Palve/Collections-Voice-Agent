"""Audio prep tests for STT."""

from utils import pcm_to_wav, prepare_pcm_for_stt, resample_pcm16


def test_prepare_pcm_resamples():
    """48kHz buffer is resampled to 16kHz."""
    pcm48 = b"\x00\x01" * 4800
    prepared, rate = prepare_pcm_for_stt(
        pcm48, target_rate=16000, source_rate=48000, min_ms=100
    )
    assert prepared is not None
    assert rate == 16000
    assert len(prepared) < len(pcm48)


def test_prepare_pcm_rejects_short():
    """Very short clips are rejected."""
    pcm, rate = prepare_pcm_for_stt(
        b"\x00\x00" * 100, target_rate=16000, source_rate=16000, min_ms=300
    )
    assert pcm is None
    assert rate == 0


def test_pcm_to_wav_header():
    """WAV has RIFF header."""
    wav = pcm_to_wav(b"\x00\x00" * 1600, 16000)
    assert wav[:4] == b"RIFF"
