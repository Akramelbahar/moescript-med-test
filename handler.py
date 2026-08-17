import os
import time
import base64
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod
import torch

from muscriptor import TranscriptionModel


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_SIZE = os.getenv("MUSCRIPTOR_MODEL", "large")

MAX_AUDIO_SIZE_MB = int(
    os.getenv("MAX_AUDIO_SIZE_MB", "150")
)

MAX_AUDIO_BYTES = (
    MAX_AUDIO_SIZE_MB * 1024 * 1024
)


# =========================================================
# LOAD MUSCRIPTOR ONCE
# =========================================================

print("=" * 60)
print("Starting MuScriptor RunPod worker")
print(f"Model: {MODEL_SIZE}")
print("=" * 60)


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available"
    )


print(
    f"GPU: {torch.cuda.get_device_name(0)}"
)


load_start = time.perf_counter()


model = TranscriptionModel.load_model(
    MODEL_SIZE,
    device="cuda"
)


print(
    f"MuScriptor {MODEL_SIZE} loaded in "
    f"{time.perf_counter() - load_start:.2f}s"
)


# =========================================================
# DOWNLOAD AUDIO
# =========================================================

def download_audio(url, destination):

    parsed = urlparse(url)

    if parsed.scheme not in (
        "http",
        "https"
    ):
        raise ValueError(
            "audio_url must use HTTP or HTTPS"
        )

    downloaded_bytes = 0

    with requests.get(
        url,
        stream=True,
        timeout=(15, 300)
    ) as response:

        response.raise_for_status()

        content_length = (
            response.headers.get(
                "Content-Length"
            )
        )

        if content_length:

            if int(content_length) > MAX_AUDIO_BYTES:

                raise ValueError(
                    f"Audio exceeds "
                    f"{MAX_AUDIO_SIZE_MB} MB"
                )

        with open(destination, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                downloaded_bytes += len(chunk)

                if (
                    downloaded_bytes
                    > MAX_AUDIO_BYTES
                ):
                    raise ValueError(
                        f"Audio exceeds "
                        f"{MAX_AUDIO_SIZE_MB} MB"
                    )

                f.write(chunk)

    return downloaded_bytes


# =========================================================
# CONVERT ANY AUDIO -> WAV
# =========================================================

def convert_to_wav(
    input_path,
    output_path
):

    print(
        f"Converting audio to WAV..."
    )

    command = [
        "ffmpeg",

        # Don't print unnecessary output
        "-hide_banner",
        "-loglevel",
        "error",

        # Replace output if it exists
        "-y",

        # Input
        "-i",
        str(input_path),

        # Ignore video streams
        "-vn",

        # Standard uncompressed PCM WAV
        "-c:a",
        "pcm_s16le",

        # Output
        str(output_path)
    ]


    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg conversion failed: "
            + result.stderr
        )


    if not output_path.exists():

        raise RuntimeError(
            "FFmpeg did not create WAV file"
        )


    if output_path.stat().st_size == 0:

        raise RuntimeError(
            "Converted WAV is empty"
        )


    print(
        f"WAV created: "
        f"{output_path.stat().st_size / 1024 / 1024:.2f} MB"
    )


# =========================================================
# RUNPOD HANDLER
# =========================================================

def handler(job):

    total_start = time.perf_counter()

    print("=" * 60)
    print(
        f"Job: {job.get('id', 'unknown')}"
    )
    print("=" * 60)


    job_input = job.get(
        "input",
        {}
    )


    audio_url = job_input.get(
        "audio_url"
    )


    if not audio_url:

        return {
            "success": False,
            "error":
                "input.audio_url is required"
        }


    instruments = job_input.get(
        "instruments"
    )


    try:

        with tempfile.TemporaryDirectory() as temp:

            temp = Path(temp)


            # We deliberately don't depend on
            # the filename extension.
            #
            # FFmpeg detects the actual codec.

            original_audio = (
                temp / "input_audio"
            )

            normalized_wav = (
                temp / "input.wav"
            )


            # =================================================
            # STEP 1 — DOWNLOAD
            # =================================================

            runpod.serverless.progress_update(
                job,
                "Downloading audio"
            )


            download_start = (
                time.perf_counter()
            )


            audio_size = download_audio(
                audio_url,
                original_audio
            )


            download_seconds = (
                time.perf_counter()
                - download_start
            )


            print(
                f"Downloaded "
                f"{audio_size / 1024 / 1024:.2f} MB "
                f"in {download_seconds:.2f}s"
            )


            # =================================================
            # STEP 2 — NORMALIZE TO WAV
            # =================================================

            runpod.serverless.progress_update(
                job,
                "Converting audio to WAV"
            )


            conversion_start = (
                time.perf_counter()
            )


            convert_to_wav(
                original_audio,
                normalized_wav
            )


            conversion_seconds = (
                time.perf_counter()
                - conversion_start
            )


            print(
                f"Conversion completed in "
                f"{conversion_seconds:.2f}s"
            )


            # =================================================
            # STEP 3 — MUSCRIPTOR
            # =================================================

            runpod.serverless.progress_update(
                job,
                "Running MuScriptor transcription"
            )


            transcription_start = (
                time.perf_counter()
            )


            if instruments:

                midi_bytes = (
                    model.transcribe_to_midi(
                        normalized_wav,
                        instruments=instruments
                    )
                )

            else:

                midi_bytes = (
                    model.transcribe_to_midi(
                        normalized_wav
                    )
                )


            transcription_seconds = (
                time.perf_counter()
                - transcription_start
            )


            # =================================================
            # STEP 4 — MIDI -> BASE64
            # =================================================

            midi_base64 = (
                base64.b64encode(
                    midi_bytes
                ).decode("ascii")
            )


            total_seconds = (
                time.perf_counter()
                - total_start
            )


            print(
                f"Transcription: "
                f"{transcription_seconds:.2f}s"
            )

            print(
                f"Total job: "
                f"{total_seconds:.2f}s"
            )


            return {

                "success": True,

                "model":
                    MODEL_SIZE,

                "audio_size_bytes":
                    audio_size,

                "wav_size_bytes":
                    normalized_wav.stat().st_size,

                "midi_size_bytes":
                    len(midi_bytes),

                "download_seconds":
                    round(
                        download_seconds,
                        2
                    ),

                "conversion_seconds":
                    round(
                        conversion_seconds,
                        2
                    ),

                "transcription_seconds":
                    round(
                        transcription_seconds,
                        2
                    ),

                "total_seconds":
                    round(
                        total_seconds,
                        2
                    ),

                "midi_base64":
                    midi_base64
            }


    except Exception as error:

        print(
            f"ERROR: "
            f"{type(error).__name__}: "
            f"{error}"
        )


        return {

            "success": False,

            "error_type":
                type(error).__name__,

            "error":
                str(error)
        }


# =========================================================
# RUNPOD
# =========================================================

runpod.serverless.start({"handler": handler})
