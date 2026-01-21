import os
import subprocess
from django.conf import settings


def handle_voice_swap(input_song_path, model_path, index_path, output_dir, device_type='cpu'):
    """
    1. Separate Vocals (Demucs)
    2. Swap Voice (RVC) -> Uses 'device_type' (cpu/gpu)
    3. Merge (FFmpeg)
    """

    # --- 1. SEPARATION (Demucs) ---
    print(f">>> Step 1: Separating Vocals for {os.path.basename(input_song_path)}...")

    # Check if files already exist to save time (optional, but good for testing)
    song_name = os.path.splitext(os.path.basename(input_song_path))[0]
    separated_folder = os.path.join(output_dir, "htdemucs_ft", song_name)
    vocals_path = os.path.join(separated_folder, "vocals.wav")
    instrumental_path = os.path.join(separated_folder, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        subprocess.run(["demucs", "-n", "htdemucs_ft", input_song_path, "-o", output_dir], check=True)

    if not os.path.exists(vocals_path):
        raise Exception("Demucs failed to separate audio.")

    # --- 2. INFERENCE (RVC Voice Swap) ---
    print(f">>> Step 2: Swapping Voice using {device_type.upper()}...")

    converted_vocals_path = os.path.join(output_dir, f"{song_name}_ai_vocals.wav")

    base_dir = settings.BASE_DIR
    rvc_root = os.path.join(base_dir, "rvc_core")
    rvc_script = os.path.join("tools", "infer_cli.py")

    # --- CPU VS GPU LOGIC ---
    if device_type == 'gpu':
        rvc_device = "cuda:0"
        is_half = "True"  # GPU likes half precision (faster)
    else:
        rvc_device = "cpu"
        is_half = "False"  # CPU needs float precision (slower but works)

    command = [
        "python", rvc_script,
        "--input_path", vocals_path,
        "--opt_path", converted_vocals_path,
        "--model_name", f"{model_path}.pth",
        "--index_path", index_path,
        "--f0_up_key", "0",
        "--f0_method", "rmvpe",
        "--device", rvc_device,
        "--is_half", is_half,
        "--filter_radius", "3",
        "--resample_sr", "0",
        "--rms_mix_rate", "0.25",
        "--protect", "0.33"
    ]

    print(f"Running RVC command: {' '.join(command)}")
    subprocess.run(command, cwd=rvc_root, check=True)

    if not os.path.exists(converted_vocals_path):
        raise Exception("RVC Inference failed. Output file not created.")

    # --- 3. MERGING (FFmpeg) ---
    print(">>> Step 3: Mixing Final Track...")
    final_output = os.path.join(output_dir, f"AI_{song_name}.mp3")

    subprocess.run([
        "ffmpeg", "-i", instrumental_path, "-i", converted_vocals_path,
        "-filter_complex", "amix=inputs=2:duration=first:dropout_transition=2",
        "-b:a", "320k",
        final_output, "-y"
    ], check=True)

    return final_output