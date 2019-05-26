nohup python -m cProfile -o profiling_results process_file_multithreaded.py
wait
nohup python -m cProfile -o profiling_results_main main.py
wait
nohup python convert_to_torch.py

