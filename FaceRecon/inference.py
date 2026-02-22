"""
Legacy entrypoint.

Use the new hackathon pipeline scripts:
1) py compile_models.py
2) py run_recognition.py --mode demo
3) py run_recognition.py --mode enroll --name "alice" --input-npy sample_1.npy
4) py run_recognition.py --mode search --input-npy sample_2.npy
"""

if __name__ == "__main__":
    print("Use compile_models.py and run_recognition.py for the full pipeline.")