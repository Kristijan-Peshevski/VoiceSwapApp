from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .utils import handle_voice_swap
import os
import shutil


def get_available_models():
    """Scans weights folder for available .pth models."""
    weights_dir = os.path.join(settings.BASE_DIR, 'rvc_core', 'assets', 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        return []
    # Return list of filenames without .pth extension
    return [f.replace('.pth', '') for f in os.listdir(weights_dir) if f.endswith('.pth')]


def index(request):
    """Render home page with list of models."""
    models = get_available_models()
    return render(request, 'index.html', {'models': models})


def process_audio(request):
    models = get_available_models()

    if request.method == 'POST' and request.FILES.get('song'):
        song_file = request.FILES['song']
        selected_model = request.POST.get('model_name')
        selected_device = request.POST.get('device_type')  # 'cpu' or 'gpu'

        if not selected_model:
            return render(request, 'index.html', {'models': models, 'error': "Please select a voice model."})

        # Save File
        fs = FileSystemStorage()
        filename = fs.save(song_file.name, song_file)
        uploaded_file_path = fs.path(filename)

        # Paths
        MODEL_NAME = selected_model
        # Try to find specific index, otherwise ignore (optional)
        INDEX_PATH = os.path.join(settings.BASE_DIR, 'models', f'{MODEL_NAME}.index')
        if not os.path.exists(INDEX_PATH):
            print(f"Warning: Index not found at {INDEX_PATH}, quality might be lower.")

        # Check Model Exists
        if not os.path.exists(os.path.join(settings.BASE_DIR, 'rvc_core', 'assets', 'weights', f"{MODEL_NAME}.pth")):
            return render(request, 'index.html', {'models': models, 'error': f"Model {MODEL_NAME}.pth not found!"})

        try:
            output_dir = os.path.dirname(uploaded_file_path)

            # RUN LOGIC with Device Choice
            output_path = handle_voice_swap(
                uploaded_file_path,
                MODEL_NAME,
                INDEX_PATH,
                output_dir,
                device_type=selected_device
            )

            result_url = fs.url(os.path.basename(output_path))
            return render(request, 'result.html', {'audio_url': result_url})

        except Exception as e:
            print(f"Error: {e}")
            return render(request, 'index.html', {'models': models, 'error': str(e)})

    return render(request, 'index.html', {'models': models, 'error': "No file uploaded."})


# --- TRAINING VIEWS ---
def train_page(request):
    return render(request, 'train.html')


def setup_training_data(request):
    if request.method == 'POST':
        model_name = request.POST.get('model_name').replace(" ", "_")
        files = request.FILES.getlist('dataset_files')

        dataset_dir = os.path.join(settings.BASE_DIR, 'rvc_core', 'datasets', model_name)

        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)

        count = 0
        for f in files:
            file_path = os.path.join(dataset_dir, f.name)
            with open(file_path, 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
            count += 1

        msg = f"✅ {count} files saved for '{model_name}'. Run 'python infer-web.py' in rvc_core to train."
        return render(request, 'train.html', {'message': msg})

    return render(request, 'train.html')